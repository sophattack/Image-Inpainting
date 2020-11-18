import tarfile
import load_data

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
print(test_data.shape)
print(train_data.shape)

###############################