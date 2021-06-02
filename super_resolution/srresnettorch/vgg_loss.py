import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):
    """
    returns features of an image from VGG19
    Used as part of VGG loss
    """

    def __init__(self):
        super(VGGLoss, self).__init__()

        vgg_net = vgg19(pretrained=True)

        self.features = nn.Sequential(*list(vgg_net.features.children())[:18])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        input: 
            Tensor
        output:
            Tensor
        """
        return self.features(img)