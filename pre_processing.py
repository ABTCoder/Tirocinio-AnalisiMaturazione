import matlab.engine
import numpy as np
import cv2 as cv
import time
import random
from skimage.draw import ellipse as fill_ellipse, ellipse_perimeter
from skimage.transform import hough_ellipse

import utils


def extract_matlab_ellipses(image, engine):
    """
    Metodo di estrazione basato sull'algoritmo in MATLAB

    :param image: L'immagine di input, assicurarsi che venga passata in RGB o GRAYSCALE e non BGR
    :param engine: L'oggetto Matlab Engine per eseguire gli script MATLAB
    :return: Maschera generata, None se fallisce
    """
    image_mat = matlab.uint8(list(image.ravel(order='F')))
    ch = 3
    if len(image.shape) == 2:
        ch = 1
    image_mat.reshape((image.shape[0], image.shape[1], ch))  # TERZO PARAMETRO = 3 SE E' A COLORI

    ellipses = engine.get_ellipses(image_mat, nargout=1)
    mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for el in ellipses:
        rr, cc = fill_ellipse(el[1], el[0], el[3], el[2], mask.shape, -el[4])  # OTTIENE I PUNTI  DELL'AREA DELL'ELLISSE
        mask[rr, cc] = 255
        cy, cx = ellipse_perimeter(int(round(el[1])), int(round(el[0])), int(round(el[3])), int(round(el[2])),
                                   shape=mask.shape, orientation=int(round(el[4])))
        if ch == 3:
            image[cy, cx] = (0, 0, 255)
        else:
            image[cy, cx] = 255

    if ellipses.size[0] == 0:
        mask = None

    return mask


def extract_cnn_mask(image, net, visualize=False):
    """
    Metodo di estrazione della maschera basato su Mask R-CNN

    :param image: L'immagine di input
    :param net: Il modello di rete caricato con cv.dnn.readNetFromTensorflow()
    :param visualize: Booleano usato per attivare la visualizzazione delle immagini
    :return: La maschera generata, completamente nera se fallisce
    """
    COLORS = open("mask-rcnn-coco/colors.txt").read().strip().split("\n")
    COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
    COLORS = np.array(COLORS, dtype="uint8")

    LABELS = open("mask-rcnn-coco/object_detection_classes_coco.txt").read().strip().split("\n")

    (H, W) = image.shape[:2]
    # Maschera che verrà restituita
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
        # extract the class ID of the detection along with the confidence
        # (i.e., probability) associated with the prediction
        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]
        # filter out weak predictions by ensuring the detected probability
        # is greater than the minimum probability
        if confidence > 0:
            # clone our original image so we can draw on it
            clone = image.copy()
            print("CONFIDENCE = " + str(confidence))
            # scale the bounding box coordinates back relative to the
            # size of the image and then compute the width and the height
            # of the bounding box
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            center_x = int(round((startX + endX) / 2))
            center_y = int(round((startY + endY) / 2))
            if utils.check_box_center(W, H, center_x, center_y):
                boxW = endX - startX
                boxH = endY - startY

                # extract the pixel-wise segmentation for the object, resize
                # the mask such that it's the same dimensions of the bounding
                # box, and then finally threshold to create a *binary* mask
                mask = masks[i, classID]
                mask = cv.resize(mask, (boxW, boxH), interpolation=cv.INTER_LANCZOS4)
                mask = (mask > 0.3)

                # extract the ROI of the image
                roi = clone[startY:endY, startX:endX]

                # check to see if are going to visualize how to extract the
                # masked region itself

                # convert the mask from a boolean to an integer mask with
                # to values: 0 or 255, then apply the mask
                visMask = (mask * 255).astype("uint8")
                instance = cv.bitwise_and(roi, roi, mask=visMask)

                # newMask a differenza di visMask ha le stesse dimensioni dell'immagine input
                newMask[startY:endY, startX:endX][mask] = 255


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

                # show the output image with the segmented instance
                if visualize:
                    cv.imshow("Segmented", instance)
                    cv.imshow("Output", clone)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

    return newMask


def detect_ellipse(image, visualize=False):
    """
    Funzione di rilevamento delle ellissi con SciKit-Image

    :param image: L'immagine di input
    :param visualize: Per scegliere di visualizzare o meno il risultato intermedio (Canny)
    :return: La maschera generata, None se fallisce
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # CONVERSIONE IN SCALA DI GRIGI

    # gray = cv.GaussianBlur(gray, (5, 5), 0)

    sigma = 0.6
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    gray = cv.Canny(gray, lower, upper)

    if visualize:
        cv.imshow("CANNY", gray)
        cv.waitKey(0)
        cv.destroyAllWindows()

    height, width = gray.shape[0:2]
    mask = np.zeros((height, width), np.uint8)
    ellipses = hough_ellipse(gray, threshold=100, accuracy=100, min_size=round(max(height, width) / 2),
                             max_size=round(min(height, width)))  # RICERCA ELLISSI
    if ellipses.size > 0:
        # ESTRAI L'ELLISSE MIGLIORE
        ellipses.sort(order='accumulator')
        # ellipses[::-1].sort()
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


def detect_contours(image):
    """
    Funzione che applica la soglia con metodo otsu

    :param image: L'immagine di input
    :return: La maschera (Immagine in scala di grigi generata dalla soglia)
    """
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # CONVERSIONE IN SCALA DI GRIGI
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    # gray = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 7)
    _, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return gray


def transform_image(image, half_size=False, double_size=False, sharpening=False):
    """
    Funzione utilizzata per effettuare alcune trasformazioni utili per migliorare il rilevamento delle olive

    :param image: L'immagine di input
    :param half_size: Per scegliere se dimezzare la dimensione dell'immagine
    :param double_size: Per scegliere se raddoppiare la dimensione dell'immagine
    :param sharpening: Per scegliere se applicare un filtro di nitidezza
    :return:
    """
    clone = image.copy()
    if half_size:
        nw = int(clone.shape[1] * 50 / 100)
        nh = int(clone.shape[0] * 50 / 100)
        dim = (nw, nh)
        clone = cv.resize(clone, dim, interpolation=cv.INTER_CUBIC)
    if double_size:
        nw = int(clone.shape[1] * 200 / 100)
        nh = int(clone.shape[0] * 200 / 100)
        dim = (nw, nh)
        clone = cv.resize(clone, dim, interpolation=cv.INTER_LANCZOS4)
    if sharpening:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        clone = cv.filter2D(clone, -1, kernel)

    cv.destroyAllWindows()
    return clone


def calc_white_percentage(mask):
    """
    FUNZIONE CHE CALCOLA LA PERCENTUALE DI PIXEL BIANCHI IN UN'IMMAGINE BIANCA-NERA

    :param mask: La maschera in scala di grigi
    :return: La percentuale calcolata
    """
    if mask is not None:
        wh = cv.countNonZero(mask)
        return (wh / mask.size) * 100
    else:
        return 0