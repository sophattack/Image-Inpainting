import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = 'data'

def get_img_paths(name):
    '''
        Returns a list of image paths which are found in name file
    '''
    ret = []
    with open(os.path.join(data_dir,name), 'r') as reader:
        ret = reader.read().splitlines()
    return ret


def load_annotations(paths):
    '''
        Returns list of annotations found in paths.
        Per source of data:
        "A subset of the images are segmented and annotated with the objects that they contain. The annotations are in LabelMe format."
    '''
    # TODO: Annotations maybe not needed for our task?
    return []


def load_normalized_img(path):
    '''
        Return the image found at path as an array-like matrix

        The returned image will be normalized to the dimensions 200x200x3 (3 color channels)

        Color channels in B G R (same as cv2)
    '''
    full_path = os.path.abspath(os.path.join(data_dir, "Images", path))
    img = cv2.imread(full_path)
    # For some reason, some images are unable to be read through cv2.. (returns NoneType)
    # Even tho the path exists (triple checked with os.path.exists(full_path))
    # Specifically I encountered this issue with "buffet/Buffet_Set_Up_gif.jpg" => Previous paths in file worked correctly
    # My guess is that maybe since it used to be a gif (per _gif), the format of the file is different?
    # Per cv2.imread docs, file type is determined by content. 
    # If undeduceable, will try to read with matplotlib.pyplot.imread which determines type by file extension
    # NOTE: matplotlib uses RGB (or RGBA) instead of BGR => Will swap axis here
    if img is None:
        img = plt.imread(full_path)
        if img.shape[2] == 4: # RGBA, drop the alpha channel
            img = img[:, :, :-1] # Dont include last channel
        # Convert from RGB to BGR
        img = img[:, :, ::-1]
    # Resize image such that the smallest axis is downsized to 200
    h, w = img.shape[0], img.shape[1]
    if h < w: # Height is shorter than width
        shrink_ratio = 200.0/h
    else: # Width is shorter than height
        shrink_ratio = 200.0/w
    new_h = int(shrink_ratio*h)
    new_w = int(shrink_ratio*w)
    resize_img = cv2.resize(img, (new_w,new_h)) # 200xW or Hx200, where W/H is greater than 200
    # Crop the center 200x200
    start_h = (new_h-200)//2
    start_w = (new_w-200)//2
    return resize_img[start_h:start_h+200, start_w:start_w+200]


def load_images(paths):
    '''
        Returns a list of images found in paths.
            The original images are all of different dimensions. 
            The dimensions will be normalized before returning
        Per source of data:
        "All images have a minimum resolution of 200 pixels in the smallest axis"
        After normalization, the images will have dimension 200x200.
        See data/README.md for more info
    '''
    images = np.zeros((len(paths), 200, 200, 3), dtype=np.uint8) # NumImagesx200x200x3 => 3 color channels
    for path_i in tqdm(range(len(paths))):
        images[path_i] = load_normalized_img(os.path.normpath(paths[path_i]))
    return images


def load_train_data():
    '''
        Returns the images and annotations for the train images
        
        :returns: (annotations, imgs)
    '''
    paths = get_img_paths("TrainImages.txt")
    # return load_annotations(paths), load_images(paths)
    return load_images(paths)


def load_test_data():
    '''
        Returns the images and annotations for the test images
                
        :returns: (annotations, imgs)
    '''
    paths = get_img_paths("TestImages.txt")
    # return load_annotations(paths), load_images(paths)
    return load_images(paths)