import cv2
import random
import numpy as np
import os
def displayImageByMatrix(title, matrix):
    cv2.imshow(title, np.uint8(matrix))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def addFixedSquareMask(img, k, num):
    """
    add mask to the image
    :param img: image, read from cv2
    :param k: mask size
    :return: x, y: origin of square mask
    :return: masked_img: masked image
    :return: label_img: label image
    """
    label_img = []
    masked_img = img.copy()
    xs = []
    ys = []

    for n in range(num):
        x = random.sample(range(1, img.shape[1] - k -1), 1)[0]
        y = random.sample(range(1, img.shape[0] - k - 1), 1)[0]
        xs.append(x)
        ys.append(y)
        label_img.append(img[x:x+k, y:y+k, :])
        masked_img = cv2.rectangle(masked_img, pt1=(x, y), pt2=(x + k-1, y + k-1), color=(0, 0, 0), thickness=-1)
    return masked_img, ys, xs, label_img

if __name__ == "__main__":
    savepath = "./save"
    imgpath = "./data/b9.jpg"
    img = cv2.imread(imgpath)
    masked_img, x, y, label_img = addFixedSquareMask(img, 10, 3)
    # print(masked_img.shape)
    # print(label_img.shape)
    print("Location: {}, {}", x, y)
    cv2.imwrite(os.path.join(savepath, os.path.basename(imgpath)), masked_img)


