import torch.nn as nn
import torch.nn.functional as F
import torch

class ShortcutSaver:
    
    def  __init__(self) -> None:
        
        self.short_cut_storage: list = []

    def place_shorcut(self, data):

        self.short_cut_storage.append(data)


class ResBlock(nn.Module):

    def __init__(self, short_cuts: ShortcutSaver, in_channels, out_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()

        self.short_cuts = short_cuts
        self.conv1 = self._add_conv(in_channels, out_channels, kernel_size, stride, padding) # input, output = filters
        # self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU()
        self.conv2 = self._add_conv(out_channels, out_channels, kernel_size, stride, padding) #out_put_ out_put


    def forward(self, x):

        short_cut = x
        self.short_cuts.place_shorcut(short_cut)

        out = self.conv1(x)
        out = self.prelu(out)
        out = self.conv2(out)

        for cut in self.short_cuts.short_cut_storage:
            out += cut

        return out


    def _add_conv(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)



class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, upsample_filters,
                up_stride, up_kernel):
        super(UpsamplingBlock, self).__init__()

        self.conv = self._add_conv(in_channels, out_channels, kernel_size, stride, padding)
        self.upsample = nn.ConvTranspose2d(in_channels= out_channels,
                                            out_channels=upsample_filters,
                                            stride=up_stride,
                                            kernel_size=up_kernel,
                                            padding=padding)
        self.prelu = nn.PReLU()


class UpsamplingBlockShuffle(nn.Module):

    def __init__(self, in_channels, kernel_size, stride, padding, factor):
        super(UpsamplingBlockShuffle, self).__init__()

        out_channels = in_channels * (factor ** 2)

        self.conv = self._add_conv(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.upsample = nn.PixelShuffle(factor)
        self.prelu = nn.PReLU()


    def forward(self, x):

        out = self.conv(x)
        out = self.upsample(out)
        out = self.prelu(out)

        return out


    def _add_conv(self, in_channels, out_channels, kernel_size, stride, padding):

        return nn.Conv2d(in_channels = in_channels, 
                        out_channels = out_channels,
                        kernel_size = kernel_size,
                        stride = stride,
                        padding = padding,
                        bias=False)



class SRResNet(nn.Module):

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

        # self.relu = nn.ReLU(inplace=True)
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
        # channels = [48, 128]
        # upsample_blocks = [UpsamplingBlock(channels[i], 128, 3, 1, 1, 128, 2, 2) 
        #                             for i in range(2)]

        # self.upsample_blocks = nn.Sequential(
        #                                     UpsamplingBlock(48, 96, 3,1,1, 96, 1, 1),
        #                                     UpsamplingBlock(96, 96, 3,1,1, 96, 2, 14))

        # self.upsample_blocks = nn.Sequential(
                                            # UpsamplingBlock(48, 96, 3,1,1, 96, 2, 10))


        self.upsample_blocks = UpsamplingBlockShuffle(32, 3,  1,  1 , 2)

        #final_conv
        self.final_conv = nn.Conv2d(in_channels = 32, 
                        out_channels = 3,
                        kernel_size = 3,
                        stride = 1,
                        padding = 1,
                        bias=False)


    def forward(self, x):

        out = self.conv0(x)
        out = self.prelu(out)

        short_cut = out
        self.short_cuts.place_shorcut(short_cut)

        out = self.res_blocks(out)

        for cut in self.short_cuts.short_cut_storage:
            out += cut

        self.short_cuts.short_cut_storage = []

        out = self.post_conv(out)
        out = self.upsample_blocks(out)
        out = self.final_conv(out)
        out = self.prelu(out)

        return out

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('Using {} device'.format(device))

# # model = SRResNet(3, 64, 3, 1, 1).to(device)
# # print(model)



# a = SRResNet(3, 64, 3, 1, 1)
# dummy = torch.ones((3, 256, 256))
# dummy = torch.unsqueeze(dummy, 0)
# print(dummy.shape)
# dummy = a(dummy)
# print(dummy.shape)
# print(torch.cuda.is_available())