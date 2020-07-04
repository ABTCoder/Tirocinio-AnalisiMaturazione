import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.transform import hough_ellipse
from skimage.draw import ellipse as fill_ellipse
from skimage.feature import canny
from skimage.util import img_as_float, img_as_ubyte


# LETTURA DEI BOUNDING BOX
def read_boxes(filename):
    bb_list = np.loadtxt(filename, int)
    converted_list = []
    # CONVERSIONE DAL FORMATO XCENTER, YCENTER, WIDTH , HEIGHT
    # A XSTART , XSTOP, YSTART, YSTOP
    for box in bb_list:
        half_width = math.floor(box[2] / 2)
        half_height = math.floor(box[3] / 2)
        xstart = box[0] - half_width
        xstop = box[0] + half_width + 1
        ystart = box[1] - half_height
        ystop = box[1] + half_height + 1
        converted_list.append([xstart, xstop, ystart, ystop])
    return converted_list


# ESTRAI I SINGOLI FRUTTI DALL'IMMAGINE IN BASE ALLE BOUNDING BOXES
def extract_histograms(filename):
    hsv_hists = []
    bgr_hists = []
    boxes = read_boxes("bboxes2.txt")
    fruits = cv.imread(filename)
    for box in boxes:
        sub_img_bgr = fruits[box[2]:box[3], box[0]:box[1]]
        sub_img_hsv = cv.cvtColor(sub_img_bgr, cv.COLOR_BGR2HSV_FULL)
        mask, skip_mask = detect_ellipse(sub_img_bgr)
        fig, ax = plt.subplots()
        if skip_mask:
            bh, gh, rh = plot_histogram(ax, sub_img_bgr, "BGR")
            hh, sh, vh = plot_histogram(ax, sub_img_hsv, "HSV")
            ax.plot(np.arange(256), hh, label="test")
            ax.legend()
            plt.show()
        else:
            bh, gh, rh = plot_histogram(ax, sub_img_bgr, "BGR", mask)
            hh, sh, vh = plot_histogram(ax, sub_img_hsv, "HSV", mask)
            ax.legend()
            plt.show()

        bgr_hists.append([bh, gh, rh])
        hsv_hists.append([hh, sh, vh])
    return hsv_hists, bgr_hists


# TEST RILEVAMENTO DEI CERCHI
def detect_ellipse(image):
    skip_mask = False
    cv.imshow("DEF", image)
    cv.waitKey(0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # gray = cv.bilateralFilter(gray, 9, 75, 75)
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    cv.imshow("BLUR", gray)
    cv.waitKey(0)
    # gray = canny(gray, sigma=6.0, low_threshold=0.3, high_threshold=0.9)
    gray = cv.Canny(gray, 90, 220)
    cv.imshow("CANNY", img_as_ubyte(gray))
    cv.waitKey(0)
    cv.destroyAllWindows()
    height, width = image.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    ellipses = hough_ellipse(gray, threshold=4, accuracy=100, min_size=round(max(height, width)/2))
    if ellipses.size > 0:
        ellipses.sort(order='accumulator')
        print(ellipses)
        best = list(ellipses[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        print(yc, xc, a, b)
        if a == 0 or b == 0:
            skip_mask = True
        else:
            rr, cc = fill_ellipse(yc, xc, a, b, mask.shape)
            mask[rr, cc] = 1
            cv.imshow("MASK", mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        skip_mask = True
    return mask, skip_mask


# CALCOLA E STAMPA L'ISTOGRAMMA PER I 3 CANALI
def plot_histogram(axis, image, label="123", mask=None):
    c1, c2, c3 = cv.split(image)
    r1 = cv.calcHist([c1], [0], None, [256], [0, 256],) #Istogramma di opencv è 10x più veloce di numpy
    axis.hist(c1.ravel(), bins=256, range=[0, 256], label=label[0])
    r2 = cv.calcHist([c2], [0], None, [256], [0, 256],)  # Istogramma di opencv è 10x più veloce di numpy
    axis.hist(c2.ravel(), bins=256, range=[0, 256], label=label[1])
    r3 = cv.calcHist([c3], [0], None, [256], [0, 256],)  # Istogramma di opencv è 10x più veloce di numpy
    axis.hist(c3.ravel(), bins=256, range=[0, 256], label=label[2])
    return r1, r2, r3


hsv, bgr = extract_histograms("olive2.jpg")

img = cv.imread("olive2.jpg")
fig1, ax1 = plt.subplots()
vals = plot_histogram(ax1, img)
plt.show()


cv.destroyAllWindows()
