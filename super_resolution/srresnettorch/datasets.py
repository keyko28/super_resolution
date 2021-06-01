import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import  List

class ImageDataset(Dataset):

    def __init__(self, path, hr_shape: list, mean: List[float], std: List[float], factor: int = 4) -> None:

        hr_height, hr_width = hr_shape

        self.files = sorted(glob.glob(path + "/*.*"))

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // factor, hr_width  // factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __getitem__(self, index : int) -> dict:

        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):

        return len(self.files)