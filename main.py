import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.transform import hough_ellipse
from skimage.draw import ellipse as fill_ellipse
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
# RITORNA DUE LISTE CONTENENTI LE FREQUENZE D'ISTOGRAMMA PER OGNI CANALE
def extract_histograms(filename, boxes_name):
    hsv_hists = []
    bgr_hists = []
    boxes = read_boxes(boxes_name)
    fruits = cv.imread(filename)
    for box in boxes:
        sub_img_bgr = fruits[box[2]:box[3], box[0]:box[1]]
        sub_img_hsv = cv.cvtColor(sub_img_bgr, cv.COLOR_BGR2HSV_FULL)

        mask = detect_ellipse(sub_img_bgr)  # METODO 1 (ELLISSI)
        # mask = detect_contours(sub_img_bgr)  # METODO 2

        fig, ax = plt.subplots()

        bh, gh, rh = plot_histogram(ax, sub_img_bgr, "BGR", mask)
        hh, sh, vh = plot_histogram(ax, sub_img_hsv, "HSV", mask)
        ax.legend()
        plt.show()

        bgr_hists.append([bh, gh, rh])
        hsv_hists.append([hh, sh, vh])
    return hsv_hists, bgr_hists


# FUNZIONE DI RILEVAMENTO DELLE ELLISSI
# RESTITUISCE LA MASCHERA , None SE NON E' STATA GENERATA
def detect_ellipse(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # CONVERSIONE IN SCALA DI GRIGI
    # gray = cv.Laplacian(gray, cv.CV_16S, ksize=3)
    # gray = cv.convertScaleAbs(gray)

    gray = cv.bilateralFilter(gray, 9, 75, 75)  # SFOCATURA
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    # gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow("BLUR", gray)
    cv.waitKey(0)

    sigma = 0.33
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    gray = cv.Canny(gray, lower, upper)

    cv.imshow("CANNY", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    height, width = image.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    ellipses = hough_ellipse(gray, threshold=200, accuracy=100, min_size=round(max(height, width)/2),
                             max_size=round(min(height, width)))  # RICERCA ELLISSI
    if ellipses.size > 0:
        # ESTRAI L'ELLISSE MIGLIORE
        ellipses.sort(order='accumulator')
        print(ellipses)
        best = list(ellipses[-1])
        yc, xc, a, b = [int(round(x)) for x in best[1:5]]
        rotation = best[5]
        print(yc, xc, a, b)
        if a == 0 or b == 0:
            mask = None
        else:
            rr, cc = fill_ellipse(yc, xc, a, b, mask.shape, rotation)  # OTTIENE L'AREA DELL'ELLISSE
            mask[rr, cc] = 255
            cv.imshow("MASK", mask)
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        mask = None
    return mask


# FUNZIONE DI RILEVAMENTO DEI CONTORNI
# RESTITUISCE LA MASCHERA
def detect_contours(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # CONVERSIONE IN SCALA DI GRIGI
    # gray = cv.Laplacian(gray, cv.CV_16S, ksize=3)
    # gray = cv.convertScaleAbs(gray)

    gray = cv.bilateralFilter(gray, 9, 75, 75)  # SFOCATURA
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    # gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow("BLUR", gray)
    cv.waitKey(0)

    sigma = 0.33
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    gray = cv.Canny(gray, lower, upper)
    cv.imshow("CANNY", gray)
    cv.waitKey(0)

    se = np.ones((7, 7), dtype='uint8')
    gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, se, iterations=2)  # CHIUDE I BUCHI DEI CONTORNI
    cv.imshow("MORP", gray)
    cv.waitKey(0)
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # mask = cv.drawContours(gray, contours, -1, 255, 3)
    mask = cv.fillPoly(gray, contours, 255)
    cv.imshow("FILL", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return mask


# CALCOLA E STAMPA L'ISTOGRAMMA PER I 3 CANALI
def plot_histogram(axis, image, label="123", mask=None):
    if mask is None:
        mask = np.full(image.shape[0:2], 255, np.uint8)
        
    c1, c2, c3 = cv.split(image)

    r1 = cv.calcHist([c1], [0], mask, [256], [0, 256])  # Istogramma di opencv è 10x più veloce di numpy
    # axis.hist(c1.ravel(), bins=256, range=[0, 256], label=label[0]) # Hist di matplotlib ricalcola le frequenze
    axis.bar(np.arange(256), r1.ravel(), label=label[0])

    r2 = cv.calcHist([c2], [0], mask, [256], [0, 256])
    axis.bar(np.arange(256), r2.ravel(), label=label[1])

    r3 = cv.calcHist([c3], [0], mask, [256], [0, 256])
    axis.bar(np.arange(256), r3.ravel(), label=label[2])

    return r1, r2, r3


hsv, bgr = extract_histograms("mele.jpg", "bboxes_mele.txt")

# img = cv.imread("olive2.jpg")
# fig1, ax1 = plt.subplots()
# vals = plot_histogram(ax1, img)
# plt.show()


cv.destroyAllWindows()
