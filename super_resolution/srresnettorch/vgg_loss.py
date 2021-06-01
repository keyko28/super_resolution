import torch.nn as nn
from torchvision.models import vgg19

class VGGLoss(nn.Module):

    def __init__(self):
        super(VGGLoss, self).__init__()

        vgg_net = vgg19(pretrained=True)

        self.features = nn.Sequential(*list(vgg_net.features.children())[:18])

    def forward(self, img):

        return self.features(img)