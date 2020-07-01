import os
import numpy as np
import cv2
from scipy.signal import medfilt2d
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.interpolation import geometric_transform
import matplotlib.pylab as plt

def spectrogram_detector(img_in, fast_output_name, aware_output_name):
    # Input
    img = img_in
    print("Img shape")
    print(img.shape)

    # Parameters
    block = 256     # cropped block size
    lam = 1         # contrast parameter
    sig = 1         # contrast parameter
    tau = 2         # contrast parameter
    dc_margin = 10  # zero-region around DC

    # low-pass filter
    alpha = np.array([(-0.25, 0.50, -0.25),
                      ( 0.50, 0,     0.50),
                      (-0.25, 0.50, -0.25)])

    # FFT contrast function
    def contrast(x):
        return cv2.GaussianBlur(x, (3, 3), 0)**3

    # Crop

    img = img[0:block, 0:block]

    # P-map computation
    e = img - cv2.filter2D(img, -1, alpha, cv2.BORDER_REFLECT)
    p = lam * np.exp(-(np.abs(e)**tau) / sig)

    # Contrasted spectrum
    P = contrast(np.abs(np.fft.fft2(p)))

    # Cumulative periodogram
    P = P[:np.int(block/2), :np.int(block/2)]
    num = np.cumsum(np.cumsum(np.abs(P)**2, axis=1), axis=0)
    den = np.sum(np.abs(P)**2)
    C = num / den



    # Remove DC peaks
    C = C[dc_margin:, dc_margin:]



    # Detect
    sobelx = cv2.Sobel(C, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(C, cv2.CV_64F, 0, 1, ksize=5)
    delta = np.max((np.abs(sobelx), np.abs(sobely)))

    # Display
    print('Confidence: {}'.format(delta))
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121), plt.imshow(P[dc_margin: , dc_margin:]), plt.title('processed pmap', color="white")
    plt.subplot(122), plt.imshow(C[dc_margin: , dc_margin:]), plt.title('detection map', color="white")
    plt.show()
    fig.savefig(fast_output_name)

    #JPEG Aware detector
    # Input
    img = img_in



    # Parameters
    block = 256     # cropped block size
    W = 4           # maximum-filter window size
    gamma = 4       # contrast enhancement
    dc_margin = 10  # zero-region around DC

    # low-pass filter
    alpha4 = np.array([(    0, 0.25,     0),
                       ( 0.25, 0,     0.25),
                       (    0, 0.25,     0)])
    alpha8 = np.array([(-0.25, 0.50, -0.25),
                       ( 0.50, 0,     0.50),
                       (-0.25, 0.50, -0.25)])

    # FFT contrast function
    def contrast(x):
        return cv2.GaussianBlur(x, (3, 3), 0)**3


    # Crop
    img = img[0:block, 0:block]


    # P-map computation
    e = img - cv2.filter2D(img, -1, alpha4, cv2.BORDER_REFLECT)
    p = np.exp(-(np.abs(e)**2))



    # Spectrum
    P = np.abs(np.fft.fft2(p))



    # Spectrum folding (custom)
    P = P[:np.int(block/2), :np.int(block/2)] + \
        np.fliplr(P[:np.int(block/2), -np.int(block/2):]) + \
        np.flipud(P[-np.int(block/2):, :np.int(block/2)]) + \
        np.flipud(np.fliplr(P[-np.int(block/2):, -np.int(block/2):]))



    # Spectrum normalization
    P_median = medfilt2d(P, (7, 7))
    P_n = P / (P_median + 1e-10)
    P_n[np.where(P_median == 0)] = 0



    # Maximum filter
    P_m_aux = maximum_filter(P_n, (W, W))
    P_m = np.zeros(P_m_aux.shape)
    P_m[np.where(P_n == P_m_aux)] = P_n[np.where(P_n == P_m_aux)]



    # Emphasize strong peaks
    P_gamma = np.max(P_m) * (P_m / np.max(P_m))**gamma



    # Cartesian to polar transformation
    def topolar(img, order=1):
        """
        Transform img to its polar coordinate representation.

        order: int, default 1
            Specify the spline interpolation order.
            High orders may be slow for large images.
        """
        # max_radius is the length of the diagonal
        # from a corner to the mid-point of img.
        max_radius = 0.5*np.linalg.norm(img.shape)

        def transform(coords):
            # Put coord[1] in the interval, [-pi, pi]
            theta = 2*np.pi*coords[1] / (img.shape[1] - 1.)

            # Then map it to the interval [0, max_radius].
            #radius = float(img.shape[0]-coords[0]) / img.shape[0] * max_radius
            radius = max_radius * coords[0] / img.shape[0]

            i = 0.5*img.shape[0] - radius*np.sin(theta)
            j = radius*np.cos(theta) + 0.5*img.shape[1]
            return i,j

        polar = geometric_transform(img, transform, order=order)

        rads = max_radius * np.linspace(0,1,img.shape[0])
        angs = np.linspace(0, 2*np.pi, img.shape[1])

        return polar, (rads, angs)



    # Detect
    P_pol, (_, _) = topolar(np.fft.fftshift(P_gamma))
    pol_sum = np.sum(P_pol, axis=1)
    delta = np.max(pol_sum) / np.median(pol_sum)


    # Display
    print('Confidence: {}'.format(delta))
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(221), plt.imshow(P[dc_margin:-dc_margin, dc_margin:-dc_margin])
    plt.subplot(222), plt.imshow(P_gamma[dc_margin:-dc_margin, dc_margin:-dc_margin])
    plt.subplot(223), plt.plot(np.linspace(0, 0.5, len(pol_sum)), pol_sum[0:])
    plt.show()
    fig.savefig(aware_output_name)
