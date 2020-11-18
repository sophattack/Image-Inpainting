from randomlyAddSquareMask import addFixedSquareMask, displayImageByMatrix
import cv2
import os
import numpy as np
import skimage
from scipy import ndimage, misc
from numpy import linalg as LA

def DFT_2D(X):
  return np.fft.fft2(X)

def IDFT_2D(X):
    return np.fft.ifft2(X).real

def ComplexConjugate(X):
    return np.conjugate(X)

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def equation_da(u_hat,alpha1, alpha2):
    M = u_hat.shape[0]
    Dx1 = np.zeros(u_hat.shape, dtype=np.complex_)
    Dy1 = np.zeros(u_hat.shape, dtype=np.complex_)
    Dx2 = np.zeros(u_hat.shape, dtype=np.complex_)
    Dy2 = np.zeros(u_hat.shape, dtype=np.complex_)
    for w1 in range(M):
        for w2 in range(M):
            Dx1[w1, w2] = ((1-np.exp(-2j*np.pi*w1/M)) ** alpha1)* np.exp(1j* np.pi * alpha1 * w1/M).astype(complex) * u_hat[w1, w2].astype(complex)
            Dy1[w1, w2] = ((1-np.exp(-2j*np.pi*w2/M)) ** alpha1)* np.exp(1j* np.pi * alpha1 * w2/M).astype(complex) * u_hat[w1, w2].astype(complex)
            Dx2[w1, w2] = ((1 - np.exp(-2j * np.pi * w1 / M)) ** alpha2) * np.exp(
                1j * np.pi * alpha2 * w1 / M) * u_hat[w1, w2].astype(complex)
            Dy2[w1, w2] = ((1 - np.exp(-2j * np.pi * w2 / M)) ** alpha2) * np.exp(
                1j * np.pi * alpha2 * w2 / M) * u_hat[w1, w2].astype(complex)

    Dx1 = IDFT_2D(Dx1)
    Dy1 = IDFT_2D(Dy1)
    Dx2 = IDFT_2D(Dx2)
    Dy2 = IDFT_2D(Dy2)
    return Dx1, Dy1, Dx2, Dy2
def hessian(x):
    """
    Calculate the hessian matrix with finite differences
    Parameters:
       - x : ndarray
    Returns:
       an array of shape (x.dim, x.ndim) + x.shape
       where the array[i, j, ...] corresponds to the second derivative x_ij
    """
    x_grad = np.gradient(x)
    hessian = np.empty((x.ndim, x.ndim) + x.shape, dtype=x.dtype)
    for k, grad_k in enumerate(x_grad):
        # iterate over dimensions
        # apply gradient again to every component of the first derivative.
        tmp_grad = np.gradient(grad_k)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[k, l, :, :] = grad_kl
    return hessian

## Function to create Gaussian windows function
def createWindowsFunction(k, sigma, x, y, ksize):
    mat = np.zeros((k,k),dtype=np.complex_)
    for i in range(k):
        for j in range(k):
            mat[i][j] = (1/(2*np.pi*sigma**2)) * np.e ** ((-1*((i-(x + ksize //2))**2+(j-(y + ksize//2))**2))/(2*sigma**2))

    # Normalization
    mat /= np.sum(mat)
    return mat

def calcK(X, alpha):
    M = X.shape[0]
    K = np.zeros(X.shape,dtype=np.complex_)
    for i in range (M):
        K[i,i] = (1- np.exp(-2j*np.pi*i/M))**alpha * np.exp(1j* np.pi * alpha * i/M)
    return K

def dc(X):
    # Apply Gaussian filter
    gm = cv2.GaussianBlur(X, (3, 3), 0.1)

    # Calculate the derivatives
    s1 = np.array([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
    s2 = np.array([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
    gx = ndimage.convolve(gm, s1, mode='nearest')
    gy = ndimage.convolve(gm, s2, mode='nearest')
    h = hessian(gm)
    print(gx)
    print(gy)
    u1 = np.divide((gx**2*h[0,0] + 2*gx*gy*h[0,1] + gy**2*h[1,1]),(gx**2+gy**2))
    u2 = np.divide((gy ** 2 * h[0, 0] - 2 * gx * gy * h[0, 1] + gx ** 2 * h[1, 1]),(gx ** 2 + gy ** 2))
    u = np.abs(np.abs(u1) - np.abs(u2))
    # print("DC: ", u)
    return u

def initialization(X, alpha1, alpha2):
    u = X
    delta_t = 0.01
    lamda = 0
    DC = np.zeros(X.shape)
    alpha =[alpha1, alpha2]
    return u, delta_t, lamda, DC, alpha



if __name__ == "__main__":
    savepath = "./save"
    imgpath = "./data/b9.jpg"
    img = cv2.imread(imgpath)
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
            masked_img_square = masked_img[:, masked_img.shape[1] - masked_img.shape[0] : ]
    print(masked_img_square.shape)

    # Steps to generate grayscale image
    grayScale = cv2.cvtColor(masked_img_square, cv2.COLOR_BGR2GRAY)

    ## Initialization
    u, delta_t, lamda, DC, alpha = initialization(grayScale, 2, 2)
    print(u[x:x + ksize, y:y + ksize])
    u_0 = DFT_2D(u)
    u_hat = u_0
    h_hat = DFT_2D(createWindowsFunction(u.shape[0], 0.1, x, y, ksize))
    Dx1, Dy1, Dx2, Dy2 = equation_da(u_hat, alpha[0], alpha[1])
    D2 = np.sqrt(Dx2 ** 2 + Dy2 ** 2 + 1e-10)
    # k = 1.4829 * np.median(np.abs(D2) - np.median(np.abs(D2)))
    k = 1
    ## Iteration
    for m in range(10):
        print("Iteration: {}".format(m+1))
        Dx1, Dy1, Dx2, Dy2 = equation_da(u_hat, alpha[0], alpha[1])
        D2 = np.sqrt(Dx2 ** 2 + Dy2 ** 2 + 1e-2)
        # print(Dx1)
        # k = 5
        print("k: {}".format(k))
        fdc = np.zeros(DC.shape)
        for i in range(DC.shape[0]):
            for j in range(DC.shape[1]):
                if np.abs(DC[i,j]) < k:
                    fdc[i,j] = 1/(1+(DC[i,j]/k)**2)
                else:
                    fdc[i,j] = 0

        lxi = fdc @ Dx1 @ u
        lyi = fdc @ Dy1 @ u
        lxn = np.linalg.inv(D2) @ Dx2 @ u
        lyn = np.linalg.inv(D2) @ Dy2 @ u
        g_hat = np.zeros(u.shape,dtype=np.complex_)
        g_hat1 = ComplexConjugate(calcK(DFT_2D(lxi), alpha[0])) + ComplexConjugate(calcK(DFT_2D(lyi), alpha[0]))
        g_hat2 = ComplexConjugate(calcK(DFT_2D(lxn), alpha[1])) + ComplexConjugate(calcK(DFT_2D(lyn), alpha[1]))
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                if i >= x and  i < x + ksize and j >= y and j  < y + ksize:
                    g_hat[i, j] = g_hat1[i, j]
                else:
                    g_hat[i, j] = g_hat2[i, j]

        u_hat = u_hat - g_hat * delta_t - lamda * h_hat @ (h_hat @ u_hat - u_0) * delta_t
        u_old = u.copy()
        u_new = IDFT_2D(u_hat)
        # u_new = cv2.GaussianBlur(u_new, (3, 3), 1.2)
        u[x:x+ksize, y:y+ksize] = u_new[x:x+ksize, y:y+ksize]
        # u = u_new
        print(u[x:x+ksize, y:y+ksize])
        # print(u)
        DC = dc(u)
        lamda = np.var(u) / np.mean(u)
        # if PSNR(u_0, u) <= PSNR(u_0, u_old):
        #     print(PSNR(u_0, u))
        #     break
        # if PSNR(u_0[x:x+ksize, y:y+ksize], u[x:x+ksize, y:y+ksize]) <= PSNR(u_0[x:x+ksize, y:y+ksize], u_old[x:x+ksize, y:y+ksize]):
        #     print(PSNR(u_0, u))
        #     break
    # new_img = cv2.cvtColor(u.astype(np.uint8).reshape(), cv2.COLOR_GRAY2RGB)
    # displayImageByMatrix("test", new_img)
    cv2.imwrite(os.path.join(savepath, "Filled_{}".format(os.path.basename(imgpath))), u)
