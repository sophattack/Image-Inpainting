import numpy as np
import os

data_dir = './data/'

def get_img_paths(name):
    '''
        Returns a list of image paths which are found in name file
    '''
    ret = []
    with open(data_dir+name, 'r') as reader:
        ret = reader.readlines()
    return ret


def load_annotations(paths):
    '''
        Returns list of annotations found in paths.
        Per source of data:
        "A subset of the images are segmented and annotated with the objects that they contain. The annotations are in LabelMe format."
    '''
    # TODO: Annotations maybe not needed for our task?
    return []


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
    # TODO: Not sure if the images should be truncated to a common size, or padded, etc etc
    return []


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