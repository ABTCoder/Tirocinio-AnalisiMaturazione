import math
import json
import numpy as np
import csv
import cv2 as cv
from sklearn.model_selection import KFold


def load_ripening_stages(datasetpath):
    stages = []
    with open(datasetpath + "maturazioni.csv") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            val = int(row[5])
            stages.append(int(val))
    return np.array(stages)


def create_masked_ripening():
    ri = load_ripening_stages()
    ms = open("masked_ripening.txt", "w")
    with open('log2.txt', 'r') as f:
        i = 0
        for line in f:
            words = line.split()
            if words[2] == "SUCCESSO,":
                ms.write(str(ri[i]) + "\n")
            i = i + 1
    ms.close()


def load_training_data(bins, colorspace, masked, dataset, three_classes=False):

    d1 = False
    d2 = False
    if dataset == 1:
        d1 = True
    if dataset == 2:
        d2 = True
    if dataset == "both":
        d1 = True
        d2 = True

    colorspace = colorspace.lower()
    if d1:
        if masked:
            x1 = np.loadtxt("dataset1/{0}_{1}bin_masked.txt".format(colorspace, bins), int)
            y1 = np.loadtxt("dataset1/masked_ripening.txt", int)
        else:
            x1 = np.loadtxt("dataset1/{0}_{1}bin.txt".format(colorspace, bins), int)
            y1 = load_ripening_stages("dataset1/")

        if three_classes:
            y1 = np.where(y1 == 2, 1, y1)
            y1 = np.where(y1 == 3, 2, y1)
            y1 = np.where(y1 >= 4, 3, y1)
    if d2:
        if masked:
            x2 = np.loadtxt("dataset2/{0}_{1}bin_masked.txt".format(colorspace, bins), int)
            y2 = np.loadtxt("dataset2/masked_ripening.txt", int)
        else:
            x2 = np.loadtxt("dataset2/{0}_{1}bin.txt".format(colorspace, bins), int)
            y2 = load_ripening_stages("dataset2/")

        if three_classes:
            y2 = np.where(y2 == 2, 1, y2)
            y2 = np.where(y2 == 3, 2, y2)
            y2 = np.where(y2 >= 4, 3, y2)

    if d1 and not d2:
        return x1, y1
    if d2 and not d1:
        return x2, y2
    if d1 and d2:
        x = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))
        return x, y


def split_data(x, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return x_train, x_test, y_train, y_test


def write_ripening_csv(filename, box_index):
    csv_file = open("maturazioni.csv", 'a')
    olives = cv.imread(filename, cv.IMREAD_UNCHANGED)
    (H, W) = olives.shape[:2]
    # boxes = read_json(box_index, H, W, 0)
    boxes = darknet_bbox(box_index, H, W)
    for box in boxes:
        print(filename, box, H, W)
        olive = olives[box[2]:box[3], box[0]:box[1]]
        cv.imshow("OLIVA", olive)
        cv.waitKey(300)
        ripening = input("Maturazione:")
        cv.destroyAllWindows()
        csv_file.write(
            filename + ", " + str(box[2]) + ", " + str(box[3]) + ", " + str(box[0]) + ", " + str(box[1]) + ", " + str(
                ripening) + "\n")
    csv_file.close()


def darknet_bbox(filename, height, width, extra=0):
    with open(filename, 'r') as f:
        vals = []
        for line in f:
            nums = line.split()
            for num in nums:
                vals.append(num)

        ndarray = np.array(vals, dtype=np.longfloat)
        ndarray = ndarray[ndarray > 0]
        k = int(ndarray.shape[0] / 4)
        ndarray = ndarray.reshape(k, 4)
        b_list = []
        for coord in ndarray:
            x_center = int(round(coord[0] * width))
            y_center = int(round(coord[1] * height))
            half_width = int(round((coord[2] * width)/2))
            half_height = int(round((coord[3] * height)/2))
            x_start = x_center - half_width
            y_start = y_center - half_height
            x_stop = x_center + half_width
            y_stop = y_center + half_height
            if extra:
                x_start = max(x_start-extra, 0)
                y_start = max(y_start-extra, 0)
                x_stop = min(x_stop + extra, width)
                y_stop = min(y_stop + extra, height)
            b_list.append([x_start, x_stop, y_start, y_stop])
    return b_list


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


def quick():
    log2 = open("log2.txt", 'w')
    with open('log.txt', 'r') as f:
        for line in f:
            words = line.split()
            if len(words) == 0 or words[0] == "Immagine:" or words[0] == "RILEVAMENTI:" or \
                    words[0] == "----------------------" or words[0] == "\n":
                continue
            else:
                log2.write(line)