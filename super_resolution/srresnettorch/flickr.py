"""
this module provides an opportunity to create a dataset
Used flicrk API

Possible sizes:
url_o: Original (4520 × 3229)
url_k: Large 2048 (2048 × 1463)
url_h: Large 1600 (1600 × 1143)
url_l=: Large 1024 (1024 × 732)
url_c: Medium 800 (800 × 572)
url_z: Medium 640 (640 × 457)
url_m: Medium 500 (500 × 357)
url_n: Small 320 (320 × 229)

One need to have a key and a secreet valid string
Also one should install flickrapi:

    pip3 install flickrapi

TODO: gather URLs and proccess them using multiorocessing

"""

from typing import Union
import flickrapi
from typing import Union, List, Any, Generator
import os
from numpy.testing._private.utils import raises
import requests
import sys
from xml.etree.ElementTree import Element


def get_photos(flickr: Any, 
               key_word: str, 
               sizes: Union[List[str], str], 
               number: int = 100) -> Generator:
    """
    this function returns a Generator instance
    which contains instance of an flickr api class
    inputs:
        :flickr - instance of an flickr api Class
        :key_word - current photo type to find
        :sizes - which size of images to look for
        :number - amount images per page
    output:
        :generator object
    """
    if isinstance(sizes, list):

        extras = ','.join(sizes)

        photos = flickr.walk(text=key_word,
                            tag_mode='all',
                            tags=key_word,
                            extras=extras,
                            per_page = number,         
                            sort='relevance')

        return photos
    
    elif isinstance(sizes, str): #str_case

        photos = flickr.walk(text=key_word,
                            tag_mode='all',
                            tags=key_word,
                            extras=sizes,
                            per_page = number,           
                            sort='relevance')

        return photos

    else:

        raise ValueError('only list or str can be passed as sizes')


def get_url(photo: Element, sizes: List[str]) -> str:
    """
    returns str representation of an image's url
    inputs:
        :photo - xml representation of a photo
        :sizes - what size we are looking for
    output:
        :url of an image
    """

    for i in range(len(sizes)):  # makes sure the loop is done in the order we want

        url = photo.get(sizes[i])

        if url:  # if url is None try with the next size

            return url


def get_urls(flickr: Any, 
            key_word: str, 
            sizes: Union[List[str], str], 
            df_amount: int = 1000) -> List[str]:
    """
    returns list of urls to form a dataset
    inputs:
        :flickr - instance of an flickr api Class
        :key_word - current photo type to find
        :sizes - what size we are looking for
        :df_amount - how many images we want to get
    output:
        :list of images urls
    """
    #take images representations
    photos = get_photos(flickr=flickr, key_word=key_word, sizes=sizes)

    counter: int = 0 # to stop later
    urls: list = []

    #this could be parallelized
    for photo in photos:

        if counter < df_amount:
            url = get_url(photo, sizes) # get single url

            if url:
                urls.append(url)
                counter += 1

        else: 
            break

    return urls


def create_folder(path: str) -> None:
    """
    creates folder in needed diractory
    inputs:
        :path - where should we create a direcotry
    """

    if not os.path.isdir(path):
        os.makedirs(path)


def download(path: str, urls: List[str]) -> None:
    """
    download images from a lis of urls
    inputs:
        :path - where to save an image
        :urls - list of gathered urls to download from
    output:
        :list of images urls
    """
    create_folder(path)

    for url in urls:

        image_name = url.split("/")[-1]
        image_path = os.path.join(path, image_name)

        if not os.path.isfile(image_path):  # ignore if already downloaded
            response=requests.get(url,stream=True)

            with open(image_path,'wb') as outfile:
                    outfile.write(response.content)


def get_dataset(flickr: Any, 
                keywords: Union[List[str], str], 
                path: str, 
                sizes: List[str], 
                df_amount: int = 1000) -> None:
    """
    main procedure. Go through key words and obtain images
    download images from a lis of urls
    inputs:
        :flickr - instance of an flickr api Class
        :key_word - current photo type to find
        :sizes - what size we are looking for
        :df_amount - how many images we want to get
    """

    for keyword in keywords:

        sys.stdout.write(f'geting urls for keyword {keyword}\n')
        urls = get_urls(flickr = flickr, key_word = keyword, sizes = sizes, df_amount = df_amount)

        sys.stdout.write(f'downloading images of {keyword} type\n')
        download(path, urls)

        sys.stdout.write(f'finished for {keyword}\n')


def main():

    key = ''
    secret = ''

    flickr=flickrapi.FlickrAPI(key, secret, cache=True)

    #specified by user
    sizes = ["url_k", "url_h", ]  
    keyword = ['tennis', 'rugby']

    path = '~\dataset'

    get_dataset(flickr = flickr,
                keywords=keyword,
                path = path,
                sizes=sizes,
                df_amount = 50)



if __name__ == "__main__":

    main()




