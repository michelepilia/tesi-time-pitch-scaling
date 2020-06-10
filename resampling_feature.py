import numpy as np
import cv2
from scipy.signal import medfilt2d
from scipy.ndimage.filters import maximum_filter


def resampling_feature2009(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)

    if img.ndim > 2:  # Keep only luminance component
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    W = 4  # maximum-filter window size
    gamma = 4  # contrast enhancement

    # low-pass filter
    alpha4 = np.array([(0, 0.25, 0),
                       (0.25, 0, 0.25),
                       (0, 0.25, 0)])

    # FFT contrast function
    def contrast(x):
        return cv2.GaussianBlur(x, (3, 3), 0) ** 3

    # P-map computation
    e = img - cv2.filter2D(img, -1, alpha4, cv2.BORDER_REFLECT)
    p = np.exp(-(np.abs(e) ** 2))

    # Spectrum
    P = np.abs(np.fft.fft2(p))

    # Spectrum normalization
    P /= np.sum(P ** 2)

    # Spectrum folding (custom)
    block = P.shape[0]
    P = P[:block // 2, :block // 2] + \
        np.fliplr(P[:block // 2, -block // 2:]) + \
        np.flipud(P[-block // 2:, :block // 2]) + \
        np.flipud(np.fliplr(P[-block // 2:, -block // 2:]))

    # Spectrum normalization
    P_median = medfilt2d(P, (7, 7))
    P_n = P / (P_median + 1e-10)
    P_n[np.where(P_median == 0)] = 0

    # Maximum filter
    P_m_aux = maximum_filter(P_n, (W, W))
    P_m = np.zeros(P_m_aux.shape)
    P_m[np.where(P_n == P_m_aux)] = P_n[np.where(P_n == P_m_aux)]

    # Emphasize strong peaks
    P_gamma = np.max(P_m) * (P_m / np.max(P_m + 1e-10)) ** gamma

    return P_gamma.flatten()


def resampling_feature2008(img: np.ndarray) -> np.ndarray:
    if img.shape[0] != img.shape[1]:
        raise ValueError('img must be square')

    img = img.astype(np.float32)

    if img.ndim > 2:  # Keep only luminance component
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114

    block = img.shape[0]
    lam = 1  # contrast parameter
    sig = 1  # contrast parameter
    tau = 2  # contrast parameter
    dc_margin = 4  # zero-region around DC

    # low-pass filter
    alpha = np.array([(-0.25, 0.50, -0.25),
                      (0.50, 0, 0.50),
                      (-0.25, 0.50, -0.25)])

    # FFT contrast function
    def contrast(x):
        return cv2.GaussianBlur(x, (3, 3), 0) ** 3

    # P-map computation
    e = img - cv2.filter2D(img, -1, alpha, cv2.BORDER_REFLECT)
    p = lam * np.exp(-(np.abs(e) ** tau) / sig)

    # Contrasted spectrum
    P = np.abs(np.fft.fft2(p))
    P = contrast(P)

    # Spectrum folding (custom)
    P = P[:np.int(block / 2), :np.int(block / 2)] + \
        np.fliplr(P[:np.int(block / 2), -np.int(block / 2):]) + \
        np.flipud(P[-np.int(block / 2):, :np.int(block / 2)]) + \
        np.flipud(np.fliplr(P[-np.int(block / 2):, -np.int(block / 2):]))
    num = np.cumsum(np.cumsum(np.abs(P) ** 2, axis=1), axis=0)
    den = np.sum(np.abs(P) ** 2)
    C = num / den

    # Remove DC peaks
    C = C[dc_margin:, dc_margin:]

    C = C.flatten()
    divisor = C.max() - C.min()
    C -= C.min()
    if divisor!=0:
        C /= divisor

    return C


def resampling_call(args):
    img = args.pop('img')
    return resampling_feature2008(img)


if __name__ == '__main__':
    img_path = 'uncompressed.tif'
    res_factor = 1.5
    jpeg_qf = 95
    block = 64
    img_in = cv2.imread(img_path, 0)
    img_res = cv2.resize(img_in, (np.int(res_factor * img_in.shape[1]),
                                  np.int(res_factor * img_in.shape[0]))).astype(np.float32)
    feat = resampling_feature2008(img_res[85:85 + block, 85:85 + block])