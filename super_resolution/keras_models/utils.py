import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path:str) -> np.ndarray:
    return np.array(Image.open(path))


def plot_sample(lr:np.ndarray, sr:np.ndarray, saveQ: bool = False) -> None:
    
    sr = sr if isinstance(sr, np.ndarray) else np.squeeze(sr.numpy(), axis=0)
    
    plt.figure(figsize=(200, 100))

    images: list = [lr, sr]
    titles: list = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
    plt.show()

    if saveQ:
        plt.savefig('sample_sr.png')
    

