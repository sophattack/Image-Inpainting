import tarfile

def untar_data(name, outdir='./data'):
    my_tar = tarfile.open('./Indoor-scene-recognition/'+name)
    my_tar.extractall(outdir)
    my_tar.close()


# Uncomment to untar data
# untar_data("indoorCVPR_09annotations.tar")
# untar_data("indoorCVPR_09.tar")
