import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


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
    images = []
    boxes = read_boxes("bboxes.txt")
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)
    for box in boxes:
        subimg = img[box[2]:box[3], box[0]:box[1]]
        subimg = detect_circle(subimg)
        images.append(cv.cvtColor(subimg, cv.COLOR_BGR2HSV))
    return images


# TEST RILEVAMENTO DEI CERCHI
def detect_circle(image):
    height, width = image.shape[0:2]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, max(height, width),
                              param1=50, param2=30, minRadius=0, maxRadius=math.floor(max(height, width) * 1.2 / 2))

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    return image


# CALCOLA E STAMPA L'ISTOGRAMMA
def plot_histogram(axis, mat, label=""):
    (n, bins) = np.histogram(mat, bins=80, density=True)  # NumPy version (no plot)
    axis.plot(.5 * (bins[1:] + bins[:-1]), n, label=label)


result = extract_hsv_images("olive.jpg")
test = result[1]
fig1, ax1 = plt.subplots()
plot_histogram(ax1, test)
plt.show()


h, s, v = cv.split(test)
fig2, ax2 = plt.subplots()
plot_histogram(ax2, h, "HUE")
plot_histogram(ax2, s, "SATURATION")
plot_histogram(ax2, v, "VALUE")
ax2.legend()
plt.show()


# test = test[:, :, 0]
for im in result:
    cv.imshow("PROVA", im)
    cv.waitKey(0)

cv.destroyAllWindows()
