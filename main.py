import matlab.engine
import random
import time
import math
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from skimage.transform import hough_ellipse
from skimage.draw import ellipse as fill_ellipse, ellipse_perimeter
from skimage.util import img_as_float, img_as_ubyte
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix


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


# LETTURA BOUNDING BOXES DAL JSON DI OLIVES FINAL
# IMPOSTARE L'EXTRA BORDER SE NECESSARIO
def read_json(n, h, w, extra=0):
    # extra = 0 per i bbox effettivi del json
    b_list = []
    with open('info.json') as f:
        data = json.load(f)

    for rect in data[n]["Label"]["Olive"]:
        min_h = h
        min_w = w
        max_h = 0
        max_w = 0
        for point in rect["geometry"]:
            min_h = min(min_h, point["y"])
            max_h = max(max_h, point["y"])
            min_w = min(min_w, point["x"])
            max_w = max(max_w, point["x"])
        xstart = max(min_w - extra, 0)
        xstop = min(max_w + extra, w)
        ystart = max(min_h - extra, 0)
        ystop = min(max_h + extra, h)
        b_list.append([xstart, xstop, ystart, ystop])

    return b_list


# ESTRAI I SINGOLI FRUTTI DALL'IMMAGINE IN BASE ALLE BOUNDING BOXES
# RITORNA DUE LISTE CONTENENTI LE FREQUENZE D'ISTOGRAMMA PER OGNI CANALE
def extract_histograms(filename, box_index, min_mask=0):
    log_file = open("log.txt", 'a')
    start_time = time.time()
    log_file.write("Immagine: " + filename + "\n")

    # hsv_hists = []
    # bgr_hists = []
    # boxes = read_boxes(boxes_name)
    fruits = cv.imread(filename)
    (H, W) = fruits.shape[:2]
    boxes = read_json(box_index, H, W, 10)
    real_boxes = read_json(box_index, H, W, 0)
    i = 0
    success = 0
    for box, r_box in zip(boxes, real_boxes):
        try:
            roi_bgr_extra = fruits[box[2]:box[3], box[0]:box[1]]
            roi_hsv_extra = cv.cvtColor(roi_bgr_extra, cv.COLOR_BGR2HSV_FULL)
            roi_bgr = fruits[r_box[2]:r_box[3], r_box[0]:r_box[1]]
            roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV_FULL)
            # cv.imwrite("pics3/input"+str(i)+".jpg", roi_bgr)

            print("DETECTING")
            # cv.imshow("input", roi_bgr_extra)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            half_size = False
            double_size = True
            sharpening = False
            for n in range(3):
                processed = pre_processing(roi_bgr_extra, half_size, double_size, sharpening)

                rgb_processed = cv.cvtColor(processed, cv.COLOR_BGR2RGB)
                mask = extract_matlab_ellipses(rgb_processed)  # METODO 4 MATLAB
                white_p = calc_white_percentage(mask)
                method = "MATLAB"

                if mask is None or white_p < min_mask:
                    print("[INFO] MATLAB first attempt failed, trying equalize Hist...")
                    eq = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
                    eq = cv.equalizeHist(eq)
                    mask = extract_matlab_ellipses(eq)
                    white_p = calc_white_percentage(mask)
                    method = "MATLAB + histequalize"

                if mask is None or white_p < min_mask:
                    print("[INFO] MATLAB failed, trying Mask R-CNN...")
                    mask = extract_cnn_mask(processed)  # METODO 3 MASK RCNN
                    white_p = calc_white_percentage(mask)
                    method = "Mask R-CNN"

                if mask is None or white_p < min_mask:
                    eq = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
                    eq = cv.equalizeHist(eq)
                    eq = cv.cvtColor(eq, cv.COLOR_GRAY2BGR)
                    mask = extract_cnn_mask(eq)
                    white_p = calc_white_percentage(mask)
                    method = "Mask R-CNN + histequalize"

                if mask is not None and white_p > min_mask:
                    break;
                sharpening = True
                if n == 1:
                    double_size = False

            if mask is None or white_p < min_mask:
                print("[INFO] All methods failed")
                mask = None
                log_file.write("Oliva " + str(i + 1) + " FALLITO\n")
            else:
                success = success + 1
                log_file.write("Oliva " + str(i + 1) + " SUCCESSO,\tPIXEL MASCHERA = " + "{:.2f}".format(white_p)
                               + "%, sharpening = " + str(sharpening) + ", double size = " + str(double_size) +
                               ", metodo = " + method + "\n")

            if double_size:
                nw = roi_bgr_extra.shape[1]
                nh = roi_bgr_extra.shape[0]
                dim = (nw, nh)
                mask = cv.resize(mask, dim, interpolation=cv.INTER_LINEAR)

            # cv.imwrite("pics3/mask" + str(i) + ".jpg", mask)

            ax = None
            # fig, ax = plt.subplots()
            if mask is not None:
                file = open("masked_ripening.txt", 'a')
                file.write(str(stag[total+i]) + "\n")
                file.close()
                for g in range(3, 6):
                    bins = pow(2, g)
                    file = open("rgb_"+str(bins)+"bin_masked.txt", 'a')
                    bh, gh, rh = plot_histogram(roi_bgr_extra, ax, "BGR", mask, bins)
                    hh, sh, vh = plot_histogram(roi_hsv_extra, ax, "HSV", mask, bins)
                    rgb_full = np.vstack((rh, gh, bh))
                    hsv_full = np.vstack((hh, sh, vh))
                    for f in rgb_full:
                        file.write(str(int(f))+" ")
                    file.write("\n")
                    file.close()
                    file = open("hsv_"+str(bins)+"bin_masked.txt", 'a')
                    for f in hsv_full:
                        file.write(str(int(f))+" ")
                    file.write("\n")
                    file.close()

            for g in range(3, 6):
                bins = pow(2, g)
                file = open("rgb_" + str(bins) + "bin.txt", 'a')
                bh, gh, rh = plot_histogram(roi_bgr, None, "BGR", None, bins)
                hh, sh, vh = plot_histogram(roi_hsv, None, "HSV", None, bins)
                rgb_full = np.vstack((rh, gh, bh))
                hsv_full = np.vstack((hh, sh, vh))
                for f in rgb_full:
                    file.write(str(int(f)) + " ")
                file.write("\n")
                file.close()
                file = open("hsv_" + str(bins) + "bin.txt", 'a')
                for f in hsv_full:
                    file.write(str(int(f)) + " ")
                file.write("\n")
                file.close()
            # ax.legend()
            # plt.show()

            # bgr_hists.append([bh, gh, rh])
            # hsv_hists.append([hh, sh, vh])
            i = i + 1
        except FileNotFoundError:
            i = i + 1
            continue

    log_file.write(
        "RILEVAMENTI: " + str(success) + " SU " + str(i) + ", TEMPO: " + str(time.time() - start_time) + "\n")
    log_file.write("----------------------\n\n")
    log_file.close()
    return success, i


# FUNZIONE DI RILEVAMENTO DELLE ELLISSI CON SCI KIT
# RESTITUISCE LA MASCHERA , None SE NON E' STATA GENERATA
def detect_ellipse(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # CONVERSIONE IN SCALA DI GRIGI

    # gray = cv.Laplacian(gray, cv.CV_16S, ksize=3)
    # gray = cv.convertScaleAbs(gray)

    gray = cv.bilateralFilter(gray, 9, 75, 75)  # SFOCATURA
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    # gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 7)
    # gray = cv.bitwise_not(gray)
    cv.imshow("BLUR", gray)
    cv.waitKey(0)

    sigma = 0.6
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    gray = cv.Canny(gray, lower, upper)

    cv.imshow("CANNY", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    height, width = gray.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    ellipses = hough_ellipse(gray, threshold=8, accuracy=100, min_size=round(max(height, width) / 2),
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

    # gray = cv.bilateralFilter(gray, 9, 75, 75)  # SFOCATURA
    # gray = cv.GaussianBlur(gray, (5, 5), 0)
    gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 9, 7)
    # _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # gray = cv.dilate(gray, kernel, iterations=2)  # dilate
    # gray = cv.bitwise_not(gray)
    # cv.imshow("BLUR", gray)

    sigma = 0.33
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    # gray = cv.Canny(gray, lower, upper)
    # cv.imshow("CANNY", gray)

    se = np.ones((7, 7), dtype='uint8')
    # gray = cv.morphologyEx(gray, cv.MORPH_CLOSE, se, iterations=1)  # CHIUDE I BUCHI DEI CONTORNI
    # cv.imshow("MORP", gray)
    contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    height, width = gray.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    # mask = cv.drawContours(gray, hull_list, -1, 255, 3)
    mask = cv.fillPoly(mask, contours, 255)
    cv.imshow("FILL", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return mask


# CALCOLA E STAMPA L'ISTOGRAMMA PER I 3 CANALI
def plot_histogram(image, axis=None, label="123", mask=None, bins=256):
    if mask is None:
        mask = np.full(image.shape[0:2], 255, np.uint8)

    c1, c2, c3 = cv.split(image)

    r1 = cv.calcHist([c1], [0], mask, [bins], [0, 256])  # Istogramma di opencv è 10x più veloce di numpy
    # axis.hist(c1.ravel(), bins=256, range=[0, 256], label=label[0]) # Hist di matplotlib ricalcola le frequenze
    r2 = cv.calcHist([c2], [0], mask, [bins], [0, 256])
    r3 = cv.calcHist([c3], [0], mask, [bins], [0, 256])

    if axis is not None:
        axis.bar(np.arange(bins), r1.ravel(), label=label[0])
        axis.bar(np.arange(bins), r2.ravel(), label=label[1])
        axis.bar(np.arange(bins), r3.ravel(), label=label[2])

    return r1, r2, r3


def extract_cnn_mask(image):
    COLORS = open("mask-rcnn-coco/colors.txt").read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    LABELS = open("mask-rcnn-coco/object_detection_classes_coco.txt").read().strip().split("\n")

    (H, W) = image.shape[:2]
    newMask = np.zeros((H, W), np.uint8)

    # construct a blob from the input image and then perform a forward
    # pass of the Mask R-CNN, giving us (1) the bounding box  coordinates
    # of the objects in the image along with (2) the pixel-wise segmentation
    # for each specific object
    blob = cv.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    # show timing information and volume information on Mask R-CNN
    print("[INFO] Mask R-CNN took {:.6f} seconds".format(end - start))
    print("[INFO] boxes shape: {}".format(boxes.shape))
    print("[INFO] masks shape: {}".format(masks.shape))
    for i in range(0, boxes.shape[2]):
        # for i in range(1):
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > 0:
            # clone our original image so we can draw on it
            clone = image.copy()
            print("CONFIDENCE = "+str(confidence))
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            center_x = int(round((startX+endX)/2))
            center_y = int(round((startY + endY) / 2))
            if check_box_center(W, H, center_x, center_y):
                boxW = endX - startX
                boxH = endY - startY

                # extract the pixel-wise segmentation for the object, resize
                # the mask such that it's the same dimensions of the bounding
                # box, and then finally threshold to create a *binary* mask
                mask = masks[i, classID]
                mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_LANCZOS4)
                mask = (mask > 0.1)

                # extract the ROI of the image
                roi = clone[startY:endY, startX:endX]

                # check to see if are going to visualize how to extract the
                # masked region itself

                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                visMask = (mask * 255).astype("uint8")
                instance = cv.bitwise_and(roi, roi, mask=visMask)
                newMask[startY:endY, startX:endX][mask] = 255
                # cv.imshow("NEWMASK", newMask)

                # show the extracted ROI, the mask, along with the
                # segmented instance
                # cv.imshow("ROI", roi)
                # cv.imshow("Segmented", instance)

                # now, extract *only* the masked region of the ROI by passing
                # in the boolean mask array as our slice condition
                roi = roi[mask]

                # randomly select a color that will be used to visualize this
                # particular instance segmentation then create a transparent
                # overlay by blending the randomly selected color with the ROI
                color = random.choice(COLORS)
                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                # store the blended ROI in the original image
                clone[startY:endY, startX:endX][mask] = blended

                # draw the bounding box of the instance on the image
                color = [int(c) for c in color]
                cv.rectangle(clone, (startX, startY), (endX, endY), color, 2)

                # draw the predicted label and associated probability of the
                # instance segmentation on the image
                text = "{}: {:.4f}".format(LABELS[classID], confidence)
                cv.putText(clone, text, (startX, startY - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # show the output image
                # cv.imshow("Output", clone)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

    return newMask


# METODO DI ESTRAZIONE BASATO SULL'ALGORITMO IN PYTHON
# RESTITUISCE LA MASCHERA
def extract_matlab_ellipses(image):
    image_mat = matlab.uint8(list(image.ravel(order='F')))
    ch = 3
    if len(image.shape) == 2:
        ch = 1
    image_mat.reshape((image.shape[0], image.shape[1], ch))  # TERZO PARAMETRO = 3 SE E' A COLORI

    ellipses = eng.get_ellipses(image_mat, nargout=1)
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for el in ellipses:
        rr, cc = fill_ellipse(el[1], el[0], el[3], el[2], mask.shape, -el[4])  # OTTIENE L'AREA DELL'ELLISSE
        mask[rr, cc] = 255
        cy, cx = ellipse_perimeter(int(round(el[1])), int(round(el[0])), int(round(el[3])), int(round(el[2])),
                                   shape=mask.shape, orientation=int(round(el[4])))
        if ch == 3:
            image[cy, cx] = (0, 0, 255)
        else:
            image[cy, cx] = 255
        # cv.imshow("MASK", mask)
        # cv.imshow("ORIGINAL", image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    if ellipses.size[0] == 0:
        mask = None

    return mask


# FUNZIONE PER EFFETTUARE ALCUNE TRASFORMAZIONI PRE RILEVAMENTO
def pre_processing(image, half_size=False, double_size=False, sharpening=False):
    if half_size:
        nw = int(image.shape[1] * 50 / 100)
        nh = int(image.shape[0] * 50 / 100)
        dim = (nw, nh)
        image = cv.resize(image, dim, interpolation=cv.INTER_CUBIC)
    if double_size:
        nw = int(image.shape[1] * 200 / 100)
        nh = int(image.shape[0] * 200 / 100)
        dim = (nw, nh)
        image = cv.resize(image, dim, interpolation=cv.INTER_LANCZOS4)
    if sharpening:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv.filter2D(image, -1, kernel)

    cv.destroyAllWindows()
    return image


def calc_white_percentage(mask):
    if mask is not None:
        wh = cv.countNonZero(mask)
        return (wh / mask.size) * 100
    else:
        return 0


def check_box_center(w, h, x, y):
    allowed_h = int(round(0.30 * h))
    allowed_w = int(round(0.30 * w))
    sx = int(round((w-allowed_w)/2))
    sy = int(round((h - allowed_h) / 2))
    ex = sx + allowed_w
    ey = sy + allowed_h

    if ex > x > sx and ey > y > sy:
        return True
    else:
        return False


def write_ripening_csv(filename, box_index):
    csv_file = open("maturazioni.csv", 'a')
    olives = cv.imread("images/"+filename, cv.IMREAD_UNCHANGED)
    (H, W) = olives.shape[:2]
    boxes = read_json(box_index, H, W, 0)
    for box in boxes:
        olive = olives[box[2]:box[3], box[0]:box[1]]
        cv.imshow("OLIVA", olive)
        cv.waitKey(300)
        ripening = input("Maturazione:")
        cv.destroyAllWindows()
        csv_file.write(filename + ", "+str(box[2])+", "+str(box[3])+", "+str(box[0])+", "+str(box[1])+", "+str(ripening)+"\n")
    csv_file.close()


def load_ripening_stages():
    stages = []
    with open("maturazioni.csv") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            stages.append(int(row[5]))
    return stages


def load_training_data(bins, colorspace, masked):
    colorspace = colorspace.lower()
    if masked:
        x = np.loadtxt(colorspace+"_"+str(bins)+"bin_masked.txt", int)
    else:
        x = np.loadtxt(colorspace + "_" + str(bins) + "bin.txt", int)
    y = np.loadtxt("masked_ripening.txt", int)
    # y = y.reshape(-1, 1)
    return x, y


def create_masked_ripening():
    ms = open("masked_ripening.txt", "w")
    with open('log2.txt', 'r') as f:
        i = 0
        for line in f:
            words = line.split()
            if words[2] == "SUCCESSO,":
                ms.write(str(stag[i])+"\n")
            i = i + 1
    ms.close()


stag = load_ripening_stages()
# print(stag)
print("[INFO] loading Mask R-CNN from disk...")
net = cv.dnn.readNetFromTensorflow("mask-rcnn-coco/frozen_inference_graph.pb",
                                   "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
print("[INFO] loading MATLAB engine...")
eng = matlab.engine.start_matlab()
print("[INFO] done loading MATLAB")


# CAMBIARE QUESTA i PER SELEZIONARE LE DIVERSE FOTO IN IMAGES
# VEDERE 23, 25, 35, 49, 1, 12 (problematico), 60, 29 troppo piccolo, 61 troppo piccolo e troppe ombre
# k = 23
successes = 0
total = 0
o = 0
# for k in range(70):
# s, t = extract_histograms("images/" + str(k+1) + ".jpg", k, min_mask=20)
# successes = successes + s
# total = total + t

# percent = (successes / total) * 100
# result = open("result.txt", 'a')
# result.write(str(successes)+" SU "+str(total)+" SUCCESSI, {:.2f}".format(percent)+"%\n")
# result.close()


# for i in range(70):
    # write_ripening_csv(str(i+1) + ".jpg", i)

print("[INFO] loading training data...")
x, y = load_training_data(8, "rgb", True)
clf = tree.DecisionTreeClassifier()
print("[INFO] fitting Tree...")
clf = clf.fit(x, y)
print("[INFO] done fitting tree")
tree.plot_tree(clf)
res = clf.predict(x[59].reshape(1, -1))
print(res)
disp = plot_confusion_matrix(clf, x, y)
print(disp.confusion_matrix)
plt.show()

