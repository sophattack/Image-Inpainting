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
    
    filters = np.random.rand(N_f, 3, 3) # N_f number of 3x3 filters
    con_desc = np.zeros((N_f, d, d, 3))
    blur_img = cv2.GaussianBlur(img,(5,5),0)
    
    for f_i in range(N_f):
        filt = filters[f_i]
        con_desc[f_i] = cv2.filter2D(blur_img, -1, filt) # dxd
    
    con_desc = np.moveaxis(con_desc, 0, -1) # dxdxN_f
    num_blocks = d//block_size
    half_block = block_size//2

    context_descriptors = np.zeros((num_blocks, num_blocks, 3, N_f))
    for col in range(num_blocks):
        block_col = (col * block_size) + half_block
        for row in range(num_blocks):
            block_row = (row * block_size) + half_block
            context_descriptors[col, row, :] = con_desc[block_col, block_row]
    
    return context_descriptors


def energy_minimization():
    '''
        The MRF part of the algorithm
    '''
    pass

def get_similar_patch_locations(patch_loc_col, patch_loc_row, context_descriptors, block_size, threshold=100.0):
    '''
        Returns a list of patch locations where the patches are contextually similar to 
        the given patch at patch_loc_col/row. 
    '''
    num_blocks = context_descriptors.shape[0]
    c_l = context_descriptors[patch_loc_col, patch_loc_row]
    
    locations = []
    for col in range(num_blocks):
        for row in range(num_blocks):
            c_m = context_descriptors[col, row] # N_f
            diss = measure_contextual_dissimilarity(c_l, c_m)
            print(diss)
            if diss <= threshold:
                locations.append((col, row))
    
    return locations

if __name__ == '__main__':
    test_data = load_data.load_test_data()
    train_data = load_data.load_train_data()

    img = train_data[0] # 200x200x3
    d = img.shape[0]
    mask = np.zeros((d, d), dtype=np.bool)
    mask[79:152, 55:78] = True

    img_cpy = np.copy(img)
    img_cpy[mask.nonzero()] = 0
    block_size = 5
    num_blocks = d//block_size
    half_block = block_size//2
    for col in range(1, num_blocks):
        img_col = col*block_size
        img_cpy[img_col, :] = [255, 0, 0]
        img_cpy[:, img_col] = [255, 0, 0]
    
    # cv2.imshow("Img with mask", img_cpy)
    # cv2.waitKey(0)
    
    # Locations of blocks that have at least one pixel needing to be filled
    patch_locations = []
    for col in range(num_blocks):
        img_col_start = col*block_size
        img_col_end = img_col_start + block_size
        for row in range(num_blocks):
            img_row_start = row*block_size
            img_row_end = img_row_start + block_size
            mask_patch = mask[img_col_start:img_col_end, img_row_start:img_row_end]
            if np.any(mask_patch):
                block_col = col + half_block
                block_row = row + half_block
                patch_locations.append((block_col, block_row))
    
    context_descriptors = get_context_descriptors(img, block_size) # num_blocks x num_blocks x N_f
    for (block_col, block_row) in patch_locations:
        similar_patches = get_similar_patch_locations(block_col, block_row, context_descriptors, block_size)
        quit()
    # Patch selection only done for patches who have altleast a pixel of "target"
    # Aka, no need to patch select if the given block is already filled
    context_aware_patch_selection(img, mask, False)