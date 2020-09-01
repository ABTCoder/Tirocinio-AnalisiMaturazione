import math
import json
import numpy as np
import csv
import cv2 as cv
from sklearn.model_selection import KFold


def write_ripening_csv(file, box):
    """
    Funzione per scrivere manualmente in un file csv le maturazioni e le bounding box delle olive, utilizzarla
    all'interno di un ciclo for

    :param file: Percorso dell'immagine
    :param box: Percorso della label o indice nel caso di lettura del json
    """
    csv_file = open("generated/maturazioni.csv", 'a')
    olives = cv.imread(file, cv.IMREAD_UNCHANGED)
    (H, W) = olives.shape[:2]
    # boxes = read_json(box_index, H, W, 0)
    boxes = darknet_bbox(box, H, W)
    for box in boxes:
        print(file, box, H, W)
        olive = olives[box[2]:box[3], box[0]:box[1]]
        cv.imshow("OLIVA", olive)
        cv.waitKey(300)
        ripening = input("Maturazione:")
        cv.destroyAllWindows()
        csv_file.write("{0}, {1}, {2}, {3}, {4}, {5}\n".format(file, box[2], box[3], box[0], box[1], ripening))


def load_ripening_stages(datasetpath):
    """
    Restituisce un ndarray nx1 con tutti i gradi di maturazioni presenti nel file maturazioni.csv, funzione usata
    all'interno di load_training_data()

    :param datasetpath: Percorso della cartella contenente maturazioni.csv
    :return: L'ndarray dei gradi di maturazione
    """
    stages = []
    with open(datasetpath + "maturazioni.csv") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            val = int(row[5])
            stages.append(int(val))
    return np.array(stages)


def time_log():
    """
    Funzione per estrarre solo i tempi di esecuzione dal file di log principale
    """
    tot = 0
    tottime = 0
    log3 = open("generated/log3.txt", 'w')
    with open('generated/log.txt', 'r') as f:
        for line in f:
            words = line.split()
            if len(words) != 0 and words[0] == "Oliva":
                tot = tot + 1
            if len(words) != 0 and words[0] == "RILEVAMENTI:":
                tottime = tottime + float(words[5])
                log3.write(line)
    avtime = tottime / tot
    log3.write("\n OLIVE: {0} TEMPO: {1} MEDIA: {2}".format(tot, tottime, avtime))


def simple_log():
    """
    Funzione per semplificare il file di log eliminando le righe vuote, trattini e info generali lasciando solamente i
    rilevamenti singoli, usata dalla funzione create_masked_ripening()
    """
    log2 = open("generated/log2.txt", 'w')
    with open('generated/log.txt', 'r') as f:
        for line in f:
            words = line.split()
            if len(words) == 0 or words[0] == "Immagine:" or words[0] == "RILEVAMENTI:" or \
                    words[0] == "----------------------" or words[0] == "\n":
                continue
            else:
                log2.write(line)


def create_masked_ripening():
    """
    Una volta creato il dataset con file di log incluso e definite manualmente le maturazioni utilizzare questa funzione
    per generare un file contenente i gradi di maturazione solo per le olive per cui è stata generata la maschera
    """
    simple_log()
    ri = load_ripening_stages()
    ms = open("generated/masked_ripening.txt", "w")
    with open('generated/log2.txt', 'r') as f:
        i = 0
        for line in f:
            words = line.split()
            if words[2] == "SUCCESSO,":
                ms.write(str(ri[i]) + "\n")
            i = i + 1
    ms.close()


def load_training_data(bins, colorspace, masked, datasets, three_classes=False):
    """
    Funzione di caricamento degli istogrammi (input) e delle label(output), si possono caricare più dataset insieme e
    concatenarli

    :param bins: Bins del dataset
    :param colorspace: Spazio di colori del dataset
    :param masked: Con o senza maschera
    :param datasets: Array contenente i percorsi dei dataset, se si passano più percorsi i dataset verranno concatenati
    insieme
    :param three_classes: Per convertire le 5 classi in 3 classi (1,2 = 1 | 3 = 2 | 4,5 = 3)
    :return: Restituisce l'ndarray di input e le label
    """

    assert isinstance(datasets, list), "Il parametro dataset deve essere un array, passare il percorso/i all'interno " \
                                       "di parentesi quadre"
    inputs = np.empty((0, bins*3))
    outputs = np.array([])
    colorspace = colorspace.lower()

    for path in datasets:
        if masked:
            x = np.loadtxt(path+"{0}_{1}bin_masked.txt".format(colorspace, bins), int)
            y = np.loadtxt(path+"masked_ripening.txt", int)
        else:
            x = np.loadtxt(path+"{0}_{1}bin.txt".format(colorspace, bins), int)
            y = load_ripening_stages("dataset1/")

        if three_classes:
            y = np.where(y == 2, 1, y)
            y = np.where(y == 3, 2, y)
            y = np.where(y >= 4, 3, y)

        inputs = np.concatenate([inputs, x])
        outputs = np.concatenate([outputs, y])

    return inputs, outputs


def split_data(x, y):
    """
    Funzione per dividere il dataset in due porzioni una per l'allenamento ed uno per il test con proporzione 80-20

    :param x: L'ndarray di dati di input
    :param y: L'ndarray di dati di output
    :return: In ordine input di allenamento, input di test, label di allenamento, label di test
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=3)
    for train_index, test_index in kf.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return x_train, x_test, y_train, y_test


def darknet_bbox(file, height, width, padding=0):
    """
    Funzione per caricare le bounding box in formato DarkNet

    :param file: Percorso del file label in formato txt
    :param height: Altezza dell'immagine, per applicare il padding senza superare i bordi
    :param width: Larghezza dell'immagine, per applicare il padding senza superare i bordi
    :param padding: Padding per aumentare la grandezza del bounding box
    :return: Lista contenente i bounding box, i valori di ogni riga in ordine sono x_start, x_stop, y_start, y_stop
    """
    with open(file, 'r') as f:
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
            if padding:
                x_start = max(x_start-padding, 0)
                y_start = max(y_start-padding, 0)
                x_stop = min(x_stop + padding, width)
                y_stop = min(y_stop + padding, height)
            b_list.append([x_start, x_stop, y_start, y_stop])
    return b_list


def read_json(n, h, w, padding=0):
    """
    Altra funzione per la lettura delle bounding box (vedere il file info.json)

    :param n: Indice del jsonArray
    :param h: Altezza dell'immagine, per applicare il padding senza superare i bordi
    :param w: Larghezza dell'immagine, per applicare il padding senza superare i bordi
    :param padding: Padding per aumentare la grandezza del bounding box
    :return: Lista contenente i bounding box, i valori di ogni riga in ordine sono x_start, x_stop, y_start, y_stop
    """
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
        xstart = max(min_w - padding, 0)
        xstop = min(max_w + padding, w)
        ystart = max(min_h - padding, 0)
        ystop = min(max_h + padding, h)
        b_list.append([xstart, xstop, ystart, ystop])

    return b_list


def read_boxes(filename):
    """
    Funzione per leggere un semplice file di bounding box creato manualmente nel formato xstart xstop ystart ystop
    (vedere i file nella cartella misc)

    :param filename: Nome o percorso del file da leggere
    :return: La lista contenente i bounding box convertiti nel formato [xstart, xstop, ystart, ystop]
    """
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


def check_box_center(w, h, x, y):
    """
    Funzione ausiliaria per il rilevamento di maschere con Mask R-CNN, serve per verificare che la maschera si trovi
    all'interno di un'area centrale all'immagine di dimensioni pari al 30% dell'originale

    :param w: Larghezza dell'immagine
    :param h: Altezza dell'immagine
    :param x: Coordinata x del centro del box che racchiude la maschera
    :param y: Coordinata y del centro del box che racchiude la maschera
    :return: True o False a seconda se la condizione è verificata
    """
    allowed_h = int(round(0.3 * h))
    allowed_w = int(round(0.3 * w))
    sx = int(round((w - allowed_w) / 2))
    sy = int(round((h - allowed_h) / 2))
    ex = sx + allowed_w
    ey = sy + allowed_h

    if ex > x > sx and ey > y > sy:
        return True
    else:
        return False


