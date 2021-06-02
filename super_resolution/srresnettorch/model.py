from typing import Union
import torch.nn as nn
import torch.nn.functional as F
import torch

class ShortcutSaver:
    """
    class provides interface to store shortcuts
    they can be used later to add it the end of the net
    also, they are released in each resblock
    """
    
    def  __init__(self) -> None:
        
        self.short_cut_storage: list = []

    def place_shorcut(self, data: torch.Tensor) -> None:
        """
        input:
            data - torch tensor to add
        """

        self.short_cut_storage.append(data)


class ResBlock(nn.Module):

    def __init__(self, 
                 short_cuts: ShortcutSaver, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: Union[int, tuple], 
                 stride: Union[int, tuple], 
                 padding: int) -> None:

        super(ResBlock, self).__init__()

        self.short_cuts = short_cuts
        self.conv1 = self._add_conv(in_channels, out_channels, kernel_size, stride, padding) # input, output = filters
        self.prelu = nn.PReLU()
        self.conv2 = self._add_conv(out_channels, out_channels, kernel_size, stride, padding) #out_put_ out_put


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        short_cut = x
        self.short_cuts.place_shorcut(short_cut)

        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)

        for cut in self.short_cuts.short_cut_storage:
            out += cut

        return out


    def _add_conv(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: Union[int, tuple], 
                 stride: Union[int, tuple], 
                 padding: int) -> nn.Conv2d:

        return nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)



class UpsamplingBlock(nn.Module):
    """
    based on ConvTranspose2d
    gives better results (1-3%)
    Need to be learned
    final model with pixel shuffle
    """

    def __init__(self, 
                in_channels: int, 
                out_channels: int, 
                kernel_size: Union[int, tuple], 
                stride: Union[int, tuple], 
                padding: int, 
                upsample_filters: int,
                up_stride: Union[int, tuple], 
                up_kernel: Union[int, tuple]) -> None:

        super(UpsamplingBlock, self).__init__()

        self.conv = self._add_conv(in_channels, out_channels, kernel_size, stride, padding)

        self.upsample = nn.ConvTranspose2d(in_channels= out_channels,
                                            out_channels=upsample_filters,
                                            stride=up_stride,
                                            kernel_size=up_kernel,
                                            padding=padding)
        self.prelu = nn.PReLU()


class UpsamplingBlockShuffle(nn.Module):
    """
    substitute for ConvTranspose2d
    works worser (1-3%)
    doesn't need to be learned (except conv layers)
    """

    def __init__(self, 
                 in_channels: int, 
                 kernel_size: Union[int, tuple], 
                 stride: Union[int, tuple], 
                 padding: int, 
                 factor: int) -> None:

        super(UpsamplingBlockShuffle, self).__init__()

        out_channels = in_channels * (factor ** 2)

        self.conv = self._add_conv(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.upsample = nn.PixelShuffle(factor)
        self.prelu = nn.PReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv(x)
        out = self.upsample(out)
        out = self.prelu(out)

        return out


    def _add_conv(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: Union[int, tuple], 
                 stride: Union[int, tuple], 
                 padding: int) -> nn.Conv2d:

        return nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)


class SRResNet(nn.Module):
    """
    class to build whole net

    """

    def __init__(self, in_channels: int = 3, 
                out_channels: int = 64, 
                kernel_size: int = 3, 
                stride: int = 1, 
                padding: int = 1,
                res_blocks: int = 4) -> None:

        super(SRResNet, self).__init__()

        #first layer
        self.conv0 = nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)

        self.prelu = nn.PReLU()

        # Residual blocks
        self.short_cuts = ShortcutSaver()
        res_blocks = [ResBlock(short_cuts = self.short_cuts, 
                               in_channels = 64, 
                               out_channels = 64, 
                               kernel_size = 3, 
                               stride = 1, 
                               padding = 1) for _ in range(res_blocks)]

        self.res_blocks = nn.Sequential(*res_blocks)

        # post res
        self.post_conv = nn.Conv2d(in_channels = 64, 
                        out_channels = 32,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)
        
        #upsample
        # self.upsample_blocks = nn.Sequential(
                                            # UpsamplingBlock(48, 96, 3,1,1, 96, 2, 10)) # in case of ConvTranspose2d usage

        self.upsample_blocks = UpsamplingBlockShuffle(32, 3,  1,  1 , 2)

        #final_conv
        self.final_conv = nn.Conv2d(in_channels = 32, 
                        out_channels = 3,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        out = self.conv0(x)
        out = self.prelu(out)

        # first shortcut
        short_cut = out 
        self.short_cuts.place_shorcut(short_cut)

        out = self.res_blocks(out)

        #add gathered shortcuts to the end
        for cut in self.short_cuts.short_cut_storage:
            out += cut

        #set variable to empty list
        self.short_cuts.short_cut_storage = []

        out = self.post_conv(out)
        out = self.upsample_blocks(out)
        out = self.final_conv(out)
        out = self.prelu(out)

        return out