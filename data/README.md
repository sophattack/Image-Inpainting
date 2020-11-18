# Directory Containing Data

Data will not be pushed to GitHub. We will maintain data each in our local repo. To start development, download the data from [here](http://web.mit.edu/torralba/www/indoor.html?fbclid=IwAR0_7QdqHvB-YT3R-ylltLE3F3Ob_tgQzPRpxi1xKNV7sYQx6cfsIuzSkXU) and save (untarred files) under "data/" dir.

See "playground.py" if unsure how to untar.

File structure in your local repo should look like:

    data/
        Annotations/
            category1/
                1a.xml
                1b.xml
                ...
            category2/
                2a.xml
                2b.xml
                ...
            ...
        Images/
            category1/
                1a.jpg
                1b.jpg
                ...
            category2/
                2a.jpg
                2b.jpg
            ...
        README.md
        TestImages.txt
        TrainImages.txt


To access the data, use the functions in load_data.py


Note that all images have different dimensions. However, per source of data, "All images have a minimum resolution of 200 pixels in the smallest axis". 
Using the data preparation technique mentioned [here](https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/), when loading the data, the images will be truncated to 200x200 (to account for the smallest possible image). 

If an image is square => A simple downsize to 200x200.
If an image is rectangle, we know the smallest axis is at least 200. Resize the rectangle such that the smallest axis is now 200. The larger dimension will be cropped in the center.

Therefore the normalized images will be 200x200x3, where the last axis denotes the color channels BGR.