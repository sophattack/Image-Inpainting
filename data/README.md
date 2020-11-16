# Directory Containing Data

Data will not be pushed to GitHub. We will maintain data each in our local repo. To start development, download the data from [here](http://web.mit.edu/torralba/www/indoor.html?fbclid=IwAR0_7QdqHvB-YT3R-ylltLE3F3Ob_tgQzPRpxi1xKNV7sYQx6cfsIuzSkXU) and save (untarred files) under "data/" dir.

See "playground.py" if unsure how to untar

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