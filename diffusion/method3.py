from randomlyAddSquareMask import addFixedSquareMask, displayImageByMatrix
import cv2
import numpy as np
import os

if __name__ == "__main__":
    # path to save
    savepath = "./save"

    # input image path
    imgpath = "./data/t1.jpg"
    img = cv2.imread(imgpath)
    ksize = 3
    num = 100
    N = ksize * 2 + 1
    # N = img.shape[0]
    masked_img, xs, ys, label_img = addFixedSquareMask(img, ksize, num)
    displayImageByMatrix("t1", masked_img)
    cv2.imwrite(os.path.join(savepath, "origin.jpg"), masked_img)
    org = masked_img.copy()
    for a in range (num*2):
        x = xs[a%num]
        y = ys[a%num]
        for n in range(N,0,-2):
            print(n)
            gm_img = np.zeros(masked_img.shape)
            print(gm_img.shape)
            for i in range(masked_img.shape[2]):
                gm_img[:,:, i] = cv2.GaussianBlur(masked_img[:,:, i], (n*2+1, n*2+1), 0.01*10**n)
            print(gm_img.shape)
            # displayImageByMatrix("t2", gm_img)
            print(masked_img.shape[0]//ksize, masked_img.shape[1]//ksize)
            new_img = cv2.resize(gm_img, (masked_img.shape[0]//n, masked_img.shape[1]//n), interpolation = cv2.INTER_NEAREST)
            # displayImageByMatrix("t3", new_img)
            # cv2.imwrite(os.path.join(savepath, "inter2_{}_{}.jpg".format(a, n)), new_img)
            k2 = min(new_img.shape[0], new_img.shape[1])
            for i in range(masked_img.shape[2]):
                # new_img[:,:, i] = cv2.GaussianBlur(new_img[:,:, i], ((((k2//4)*2)+1, ((k2//4)*2)+1)), 0.0001)
                new_img[:, :, i] = cv2.GaussianBlur(new_img[:, :, i], (3, 3),
                                                    0.001 * 10 ** (ksize//2) * 0.1*10**n)
            # cv2.imwrite(os.path.join(savepath, "g_{}_{}.jpg".format(a, n)), new_img)
                # new_img[:, :, i] = cv2.medianBlur(new_img[:, :, i], ((k2//4)*2)+1)
            print(img.shape[0], img.shape[1])
            new_img2 = cv2.resize(new_img, (img.shape[0], img.shape[1]),interpolation=cv2.INTER_LANCZOS4)
            # cv2.imwrite(os.path.join(savepath, "in2_{}_{}.jpg".format(a, n)), new_img2)
            print(masked_img[x:x+ksize, y:y+ksize])
            xmin = min(max(0, (2*x+ksize-1)//2-n//4), x)
            xmax = max(min(img.shape[0], (2*x+ksize-1)//2+n//4), x+ksize)
            ymin = min(max(0, (2*y+ksize-1)//2-n//4), y)
            ymax = max(min(img.shape[1], (2*y+ksize-1)//2+n//4), y+ksize)
            print(x,y)
            print(xmin, xmax, ymin, ymax)
            masked_img = org.copy()
            print(masked_img.shape)
            masked_img[xmin:xmax, ymin:ymax] = cv2.GaussianBlur(new_img2[xmin:xmax, ymin:ymax], (n*2+1, n*2+1), 0.001*10**n)
            # cv2.imwrite(os.path.join(savepath, "r{}.jpg".format(n)), masked_img)
        org = masked_img.copy()
    # masked_img = cv2.GaussianBlur(masked_img, (ksize*2+1,ksize*2+1), 0.1*(ksize)*np.log10(num))
    displayImageByMatrix("t4", new_img2)
    cv2.imwrite(os.path.join(savepath, "result.jpg"), masked_img)

