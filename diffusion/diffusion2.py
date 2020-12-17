from randomlyAddSquareMask import addFixedSquareMask, displayImageByMatrix
import numpy as np
import cv2
import os

def pde(N, img, x, y, k):
    alpha = 0.2
    step = 0.98
    for i in range(N):
        print("Iteration {}".format(i+1))
        for a in range (x, x + k):
            for b in range (y, y + k) :
                sec_grad = img[a, b-1] + img[a, b + 1] + img[a-1, b] + img[a+1, b] - 4 * img[a, b]
                img[a, b] = img[a, b] + alpha * sec_grad
        alpha = alpha * step
    return img



if __name__ == "__main__":
    savepath = "./save"
    imgpath = "./data/b9.jpg"
    img = cv2.imread(imgpath)
    N = 70
    ksize = 3
    masked_img, x, y, label_img = addFixedSquareMask(img, ksize)
    print(masked_img.shape)
    print(label_img.shape)
    print("Location: {}, {}", x, y)
    cv2.imwrite(os.path.join(savepath, os.path.basename(imgpath)), masked_img)

    if masked_img.shape[0] > masked_img.shape[1]:
        if x < masked_img.shape[0] // 2:
            masked_img_square = masked_img[:masked_img.shape[1], :]
        else:
            masked_img_square = masked_img[masked_img.shape[0] - masked_img.shape[1]:, :]
    else:
        if y < masked_img.shape[1] // 2:
            masked_img_square = masked_img[:, :masked_img.shape[0]]
        else:
            masked_img_square = masked_img[:, masked_img.shape[1] - masked_img.shape[0]:]
    print(masked_img_square.shape)
    grayScale = cv2.cvtColor(masked_img_square, cv2.COLOR_BGR2GRAY)
    u = pde(N, grayScale, x, y, ksize)
    cv2.imwrite(os.path.join(savepath, "Filled_{}".format(os.path.basename(imgpath))), u)