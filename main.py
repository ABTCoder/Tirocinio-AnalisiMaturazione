import math
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# img = cv.imread("mela.jpg", cv.IMREAD_UNCHANGED)
# cv.imshow("PROVA", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# matrix = np.arange(21).reshape(3, 7)
# print(matrix)
# fig, axis = plt.subplots()
# axis.plot([1,2,3,4], [4,7,8,4])
# plt.show()

# LETTURA DEI BOUNDING BOX
def read_boxes(filename):
    bb_list = np.loadtxt(filename, int)
    converted_list = []
    # CONVERSIONE DAL FORMATO XCENTER, YCENTER, WIDTH , HEIGHT
    # A XSTART , XSTOP, YSTART, YSTOP
    for box in bb_list:
        half_width = math.floor(box[2]/2)
        half_height = math.floor(box[3]/2)
        xstart = box[0] - half_width
        xstop = box[0] + half_width + 1
        ystart = box[1] - half_height
        ystop = box[1] + half_height + 1
        converted_list.append([xstart, xstop, ystart, ystop])
    return converted_list


# ESTRAI I SINGOLI FRUTTI DALL'IMMAGINE IN BASE ALLE BOUNDING BOXES
def extract_images(filename):
    images = []
    boxes = read_boxes("bboxes.txt")
    img = cv.imread(filename, cv.IMREAD_UNCHANGED)
    for box in boxes:
        subimg = img[box[2]:box[3], box[0]:box[1]]
        images.append(subimg)
    return images


cv.imshow("PROVA", extract_images("olive.jpg")[0])
cv.waitKey(0)
cv.destroyAllWindows()

