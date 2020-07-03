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
def extract_hsv_images(filename):
    hsv_images = []
    bgr_images = []
    masks = []
    boxes = read_boxes("bboxes.txt")
    img = cv.imread(filename)
    for box in boxes:
        subimg = img[box[2]:box[3], box[0]:box[1]]
        mask = detect_ellipse(subimg)
        bgr_images.append(subimg)
        hsv_images.append(cv.cvtColor(subimg, cv.COLOR_BGR2HSV_FULL))
        masks.append(mask)
    return hsv_images, bgr_images, masks


# TEST RILEVAMENTO DEI CERCHI
def detect_ellipse(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = canny(gray, sigma=4.0, low_threshold=0.3, high_threshold=0.8)
    height, width = image.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    ellipse = hough_ellipse(gray, threshold=100, min_size=math.floor(max(height, width)/2))
    params = ellipse.tolist()
    rr, cc = fill_ellipse(params[1], params[2], params[3], params[4], mask.shape)
    mask[rr, cc] = 1
    return mask


# CALCOLA E STAMPA L'ISTOGRAMMA
def plot_histogram(axis, mat, label=""):
    hist = cv.calcHist([mat], [0], None, [256], [0, 256]) #Istogramma di opencv è 10x più veloce di numpy
    axis.hist(mat.ravel(), bins=256, range=[0, 256], label=label)
    return hist


hsv, bgr, masks = extract_hsv_images("olive.jpg")
test = hsv[1]
img = cv.imread("olive.jpg")
b, g, r = cv.split(img)
fig1, ax1 = plt.subplots()
vals = plot_histogram(ax1, r)
plt.show()

# restituire anche gli istogrammi r g b
h, s, v = cv.split(test)
fig2, ax2 = plt.subplots()
plot_histogram(ax2, h, "HUE")
plot_histogram(ax2, s, "SATURATION")
plot_histogram(ax2, v, "VALUE")
ax2.legend()
plt.show()


# test = test[:, :, 0]
for im in hsv:
    cv.imshow("PROVA", im)
    cv.waitKey(0)
cv.imshow("PROVA", masks[3])
cv.waitKey(0)
cv.destroyAllWindows()
