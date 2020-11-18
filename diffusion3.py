from randomlyAddSquareMask import addFixedSquareMask, displayImageByMatrix
import numpy as np
import cv2
import os
from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage
from scipy.sparse import spdiags, diags
from scipy.sparse import identity
import skimage

## Function to compute gradient magnitude
def gradientMagnitude(img, k = 3, sigma = 0.1):

    # Apply Gaussian filter
    gm = cv2.GaussianBlur(img, (k, k), sigma)
    print("grayscale shape: ", gm.shape)

    # Calculate the gradient magnitude
    s1 = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    s2 = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    gx = ndimage.convolve(gm, s1, mode='nearest')
    gy = ndimage.convolve(gm, s2, mode='nearest')

    # steps to calculate g
    g = np.sqrt(np.power(gx, 2) + np.power(gy, 2))
    print("g shape:", g.shape)
    print("g:")
    # displayImageByMatrix("g", g)
    # cv2.imwrite(os.path.join(savepath, "g{}.jpg".format(imgIndex)), g)
    return g, gx, gy
def diffusion(img, x, y, ksize):
    delta = 0.1
    dt = 0.5
    m = img.shape[0]
    n = img.shape[1]
    u = anisotropic_diffusion(img)
    for i in range(200):
        for k in range(40):
            lap, Ix, Iy = gradientMagnitude(u)
            _, lapIx, lapIy =  gradientMagnitude(lap)

            Ix, Iy = -Iy, Ix
            lapIx = lapIx.reshape((m,n))
            lapIy = lapIy.reshape((m,n))
            Ix = Ix.reshape((m,n))
            Iy = Iy.reshape((m,n))
            dI = (lapIx * Ix) + (lapIy * Iy)
            dIpos = dI.copy()
            notdIpos = dI.copy()
            dIpos[dIpos < 0] = 0
            notdIpos[dI >= 0] = 0
            notdIpos[dI < 0] = 1
            dIpos = dIpos.reshape(m*n, 1)
            tmpArray = -1 * np.ones((m, m))
            for a in range(m):
                tmpArray[a, a] = 1
            print(tmpArray)
            d1i_forward = spdiags(tmpArray, np.array([0, 1]), m, m).toarray()
            d1j_forward = spdiags(tmpArray, np.array([0, 1]), n, n).toarray()
            tmpArray = -tmpArray
            d1i_backward = spdiags(tmpArray, np.array([-1, 0]), m, m).toarray()
            d1j_backward = spdiags(tmpArray, np.array([-1, 0]), n, n).toarray()
            d1i_forward[-1, :] = 0
            d1j_forward[-1, :] = 0
            d1i_backward[0, :] = 0
            d1j_backward[0, :] = 0
            Dif = np.kron(np.eye(n), d1i_forward)
            # print(n)
            # print(Dif.shape)
            # print(np.eye(n).shape)
            print(d1i_forward.shape)
            Djf = np.kron(d1j_forward, np.eye(m)).todense()
            Dib = np.kron(np.eye(n), d1i_backward).todense()
            Djb = np.kron(d1j_backward, np.eye(m)).todense()
            uxf = Dif*u.reshape((m*n, 1))
            uxb = Dib*u.reshape((m*n, 1))
            uyf = Djf*u.reshape((m*n, 1))
            uyb = Djb*u.reshape((m*n, 1))

            slopeLim = dIpos * np.sqrt(min(np.min(uxb), 0)**2 + max(np.max(uxf), 0) ** 2 + min(np.min(uyb), 0)**2 + max(np.max(uyf), 0)**2) + notdIpos * np.sqrt(max(np.max(uxb), 0)**2 + min(np.min(uxf), 0) ** 2 + max(np.max(uyb), 0)**2 + min(np.min(uyf), 0)**2)
            slopeLim = slopeLim.reshape(m, n)
            update = dI * slopeLim
            for a in range (x, x+ksize):
                for b in range(y, y+ksize):
                    u[a, b] = u[a, b] + dt * update[a,b]
        un = anisotropic_diffusion(u)
        for a in range (m):
            for b in range (n):
                if a >=x and a < x + ksize and b >= y and b < y+ksize:
                    u[a,b] = un[a,b]
                else:
                    u[a, b] = u[a,b]

        if np.sum(np.abs(un - u)) < delta:
            return u
    return u



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
    # Steps to generate grayscale image
    grayScale = cv2.cvtColor(masked_img_square, cv2.COLOR_BGR2GRAY)

    returnImg = diffusion(grayScale, x, y, ksize)
    cv2.imwrite(os.path.join(savepath, "Filled_{}".format(os.path.basename(imgpath))), returnImg)