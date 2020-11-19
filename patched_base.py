import load_data

def context_aware_patch_selection(img, target, adaptive):
    '''
        Returns patch selection

        :param img: input image. MxN
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
        pass


def energy_minimization():
    '''
        The MRF part of the algorithm
    '''
    pass

if __name__ == '__main__':
    test_data = load_data.load_test_data()
    train_data = load_data.load_train_data()