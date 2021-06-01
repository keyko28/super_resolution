import numpy as np
import torch
from model import SRResNet
from PIL import Image
import torchvision.transforms as transforms
from utils import UnNormalize
import os
from torchvision.utils import save_image
from typing import Union, List
from torch.autograd import Variable

class Upscaler:
    """
    To support upscaling uisng GPU device
    Your GPU should have at least 8 gb of ram
    for working with fhd images

    It is ok to use 4gb GPU to upsacle
    from <720p input

    If one doesn't have such GPU,
    CPU should be prefered
    Set 'need_cuda' to False

    Be careful with normalization
    The SRResNet has been trained used normalization
    So, a normalization parameter in save_image set to False
    But an option for playing is here

    """

    def __init__(self, model_path: str, 
                need_cuda: bool = False) -> None:

        self.model_path = model_path
        self.image = None
        
        # Tensor is needed later for pass an input to the CUDA device 
        if need_cuda and torch.cuda.is_available():
            self.cuda = True
            self.Tensor = torch.cuda.FloatTensor 

        else:
            self.cuda = False
            self.Tensor = None

    def load_model(self) -> None:
        """
        loads model
        and sets it into inference mode
        """
        
        self.model = SRResNet() # model itself
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval() # inference mode

        if self.cuda:
            self.model.cuda()

    def load_image(self, input_image_path: str, 
                    mean: Union[np.ndarray, List[float], None] = None, 
                    std: Union[np.ndarray, List[float], None] = None, 
                    unnormailzeQ: bool = False) -> None:

        """
        loads image, makes basic transfroms
        if need to UnNormalize, this process will be handeled too

        input:
        :input_image_path: str
        :mean - per channel mean value - the np.ndarray type is preferable
        :std: per channel std value - the np.ndarray type is preferable
        :unnormailzeQ - a flag to work with an unnormalization procees 
        """

        image = Image.open(input_image_path)

        image_to_tensor = transforms.ToTensor()
        self.image = image_to_tensor(image)

        if unnormailzeQ:
            unnorm = UnNormalize(mean, std)
            self.image = unnorm(self.image)

        self.image = torch.unsqueeze(self.image, 0)

        if self.cuda:
            # covert to cuda tensor type
            self.image = Variable(self.image.type(self.Tensor))
            self.image.cuda()


    def upscale_image(self, upscaled_image_path: str, 
                      upscale_image_name:str, 
                      extention: str) -> None:

        """
        cheks given extension and with no grad upscales image
        saves after all

        input:
        :upscaled_image_path - path to store upscaled image
        :upscale_image_name - image name
        :extention - one from the  ['png', 'jpg', 'tiff'
        """

        if extention not in ['png', 'jpg', 'tiff']:
            raise ValueError("the extention must be one of 'png', 'jpg', 'tiff'")

        with torch.no_grad():
            gen_hr = self.model(self.image)

        path = os.path.join(upscaled_image_path, f'{upscale_image_name}.{extention}')
        # be careful with Normalize. See info above
        save_image(gen_hr, path, normalize=False) # if True: faded, 'film-like' colours are expected 


#A basic pipeline to work with SRResNet
def main():

    path = 'D:\\pet_projects\\super_resolution\\srresnettorch\\model_weights\\srsnet_30.pth'
    image_path = 'D:\\Photo\\nyc-portra400-018.jpg'
    # image_path = 'D:\\Photo\\09052021\\test\\DSC_2049.jpg'
    # image_path = 'D:\\Photo\\09052021\\jpgs\\DSC_2052-2.jpg'
    out_path = 'D:\\Photo\\'
    name = 'test10_cuda' 

    u = Upscaler(path)
    u.load_model()
    u.load_image(image_path)
    u.upscale_image(out_path, name, 'jpg')



if __name__ == '__main__':
     main()