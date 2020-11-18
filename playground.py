import tarfile
import load_data
import cv2
import numpy as np

###############################
# Untar data
def untar_data(name, outdir='./data'):
    my_tar = tarfile.open('./Indoor-scene-recognition/'+name)
    my_tar.extractall(outdir)
    my_tar.close()

# Uncomment to untar data
# untar_data("indoorCVPR_09annotations.tar")
# untar_data("indoorCVPR_09.tar")
###############################

###############################
# Load data
test_data = load_data.load_test_data()
train_data = load_data.load_train_data()

# Show the data
print(test_data.shape)
print(train_data.shape)
train_i = np.random.choice(train_data.shape[0])
test_i = np.random.choice(test_data.shape[0])
cv2.imshow("example in train", train_data[train_i])
cv2.imshow("example in test", test_data[test_i])
cv2.waitKey(0)

###############################