import load_data
import numpy as np
import cv2

def context_aware_patch_selection(img, target, adaptive):
    '''
        Returns patch selection

        :param img: input image. MxNx3 => 200x200x3
        :param target: region to be filled. T/F matrix of size MxN. 
            img[target] returns the region to be filled.
        :param adaptive: T/F. True if adaptive partitioning will be used. 
            else simple partitioning
        :returns: patches
            patches: a list of most contextually similar blocks?
                    I think a list bc greedy wants 1, multiple candidate wants to combine them all
    '''
    if adaptive:
        # Adaptive sized block partitions
        pass
    else:
        # Fixed sized block partitions
        return fixed_sized_patch_selection(img, target)


def fixed_sized_patch_selection(img, target, block_size=5):
    '''
        Returns patch selection

        :param img: input image. MxNx3 => 200x200x3
        :param target: region to be filled. T/F matrix of size MxN. 
            img[target] returns the region to be filled.
        :param block_size: (optional) size of patch squares. 
                    Should be perfect multiple of img dimensions. Should also be odd.
        :return: (patches, block_size)
            patches: a list of locations for contextually similar patches. 
                All patches are of block_sizexblock_size. Location is center of block
                For example: [(10,10), ....., (150, 150)]
            block_size: the size of each patch. Necessary to retrieve source patch via patch location.
                loc_col, loc_row = patches[0]
                img[loc_col-block_size:loc_col+block_size, loc_row-block_size:loc_row+block_size] := patch 0
    '''
    d = img.shape[0]  # 200

    context_descriptors = get_context_descriptors(img, block_size) # num_blocks x num_blocks x N_f

    num_blocks = d/block_size

    for col in range(num_blocks):
        for row in range(num_blocks):
            c_l = context_descriptors[col, row] # N_f

    
def measure_contextual_dissimilarity(c_l, c_m):
    '''
        Returns a measure of dissimilarity between two block's contextual descriptors
        Using a distance measure
        :param c_l: N_f vector
        :param c_m: N_f vector
        :return: scalar
    '''
    diff = c_l - c_m
    return np.linalg.norm(diff)

def get_context_descriptors(img, block_size, N_f=20):
    '''
        Return context descriptors for the blocks in the image. 
        context descriptors computed using multi-channel filtering.

        :param img: input image. MxMx3 => 200x200x3
        :param block_size: size of patch squares. 
            Should be perfect multiple of img dimensions. Should also be odd.
        :param N_f: (optional) number of linear filters 
        :return: context_descriptors MxMxN_f
            context_descriptors: contextual descriptors for each block in image
                loc_col, loc_row := center location of a patch/block
                context_descriptors[loc_col, loc_row]
                    := feature vector that characterizes spatial content and textures within the block
                    -> N_f dim vector
                    -> refers to the descriptor for patch img[loc_col-block_size:loc_col+block_size, loc_row-block_size:loc_row+block_size]
    '''
    d = img.shape[0]
    
    filters = np.random.rand((N_f, 3, 3)) # N_f number of 3x3 filters
    con_desc = np.zeros((N_f, d, d))
    blur_img = cv2.GaussianBlur(img,(5,5),0)
    
    for f_i in range(N_f):
        filt = filters[f_i]
        con_desc[f_i] = cv2.filter2D(blur_img, -1, filt) # dxd
    
    con_desc = np.moveaxis(con_desc, 0, -1) # dxdxN_f
    num_blocks = d/block_size
    half_block = block_size//2

    context_descriptors = np.zeros((num_blocks, num_blocks, N_f))
    for col in range(num_blocks):
        block_col = (col * block_size) + half_block
        for row in range(num_blocks):
            block_row = (row * block_size) + half_block
            context_descriptors[col, row] = con_desc[block_col, block_row]
    
    return context_descriptors


def energy_minimization():
    '''
        The MRF part of the algorithm
    '''
    pass

if __name__ == '__main__':
    test_data = load_data.load_test_data()
    train_data = load_data.load_train_data()

    img = train_data[0] # 200x200x3
    d = img.shape[0]
    mask = np.zeros((d, d), dtype=np.bool)
    mask[30:100, 50:60] = True
    
    context_aware_patch_selection(img, mask, False)