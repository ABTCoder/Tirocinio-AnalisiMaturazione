import matlab.engine
import time
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from sklearn import tree
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import utils
import pre_processing as prp

np.set_printoptions(precision=17, floatmode="maxprec")

print("[INFO] loading Mask R-CNN from disk...")
net = cv.dnn.readNetFromTensorflow("mask-rcnn-coco/frozen_inference_graph.pb",
                                   "mask-rcnn-coco/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
print("[INFO] loading MATLAB engine...")
eng = matlab.engine.start_matlab()
print("[INFO] done loading MATLAB")


def extract_histograms(file, box, min_mask=0,  hsv=True, bins=32, plot=False, masked=True, visualize=False,
                       writelog=False, writedataset=False):
    """
    Funzione di rilevamento maschera e calcolo dell'istogramma, utilizza 2 variabili globali: net e eng inizializzati a
    Inizio script

    :param file: Percorso dell'ìmmagine
    :param box: Percorso della label o indice nel caso del json
    :param min_mask: Minima percentuale di pixel bianchi per cui la maschera è considerata OK
    :param hsv: Indica se calcolare gli istogrammi per il canale hsv
    :param bins: Numero di intervalli per il calcolo dell'istogramma
    :param plot: Attiva il tracciamento e la visualizzazione del grafico dell'istogramma
    :param masked: Indica se effettuare o meno il rilevamento della maschera e calcolare gli istogrammi su di esso
    :param visualize: Per mostrare o meno l'immagine di input e la maschera generata
    :param writelog: Per attivare o disattivare la scrittura del file di log
    :param writedataset: Per attivare o disattivare la scrittura del dataset
    :return: Ritorna nell'ordine: ndarray con gli istogrammi di tutte le olive dell'immagine, numero di successi di
    rilevamento della maschera e numero totale di olive
    """
    if writelog:
        log_file = open("log.txt", 'a')
        start_time = time.time()
        log_file.write("Immagine: " + file + "\n")

    print("[INFO] " + file)
    hists = []  # LISTA IN CUI VERRANNO MANO A MANO INSERITI I VALORI D'ISTOGRAMMA

    fruits = cv.imread(file)
    (H, W) = fruits.shape[:2]
    # boxes = utils.read_json(box, H, W, 10)  # CON PADDING
    # real_boxes = utils.read_json(box, H, W)  # SENZA PADDING
    boxes = utils.darknet_bbox(box, H, W, 10)
    real_boxes = utils.darknet_bbox(box, H, W)
    i = 0
    success = 0
    for box, r_box in zip(boxes, real_boxes):
        try:
            roi_bgr_padding = fruits[box[2]:box[3], box[0]:box[1]]
            roi_hsv_padding = cv.cvtColor(roi_bgr_padding, cv.COLOR_BGR2HSV_FULL)
            roi_bgr = fruits[r_box[2]:r_box[3], r_box[0]:r_box[1]]
            roi_hsv = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV_FULL)

            print("[INFO] DETECTING {0} IN {1}".format(i + 1, file))
            if visualize:
                cv.imshow("input", roi_bgr_padding)
                cv.waitKey(0)

            """CODICE PER GENERARE LE IMMAGINI PER RESNET, CREARE PRIMA LE MATURAZIONI E LE CARTELLE
            cv.imwrite("non_mask5/{0}/{1}.jpg".format(y1[total + i], total + i), roi_bgr_padding)
            cv.imwrite("non_mask3/{0}/{1}.jpg".format(y2[total + i], total + i), roi_bgr_padding)
            """
            mask = None
            # SE SI VUOLE GENERARE GLI ISTOGRAMMI ANCHE CON LA MASCHERA
            if masked:
                half_size = False
                double_size = True
                sharpening = False
                for n in range(3):
                    processed = prp.transform_image(roi_bgr_padding, half_size, double_size, sharpening)

                    rgb_processed = cv.cvtColor(processed, cv.COLOR_BGR2RGB)  # IMMAGINI RGB PER MATLAB
                    mask = prp.extract_matlab_ellipses(rgb_processed, eng)  # METODO 4 MATLAB
                    white_p = prp.calc_white_percentage(mask)
                    method = "MATLAB"

                    if mask is None or white_p < min_mask:
                        print("[INFO] MATLAB first attempt failed, trying equalize Hist...")
                        eq = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
                        eq = cv.equalizeHist(eq)
                        mask = prp.extract_matlab_ellipses(eq, eng)
                        white_p = prp.calc_white_percentage(mask)
                        method = "MATLAB + histequalize"

                    if mask is None or white_p < min_mask:
                        print("[INFO] MATLAB failed, trying Mask R-CNN...")
                        mask = prp.extract_cnn_mask(processed, net)  # METODO 3 MASK RCNN
                        white_p = prp.calc_white_percentage(mask)
                        method = "Mask R-CNN"

                    if mask is None or white_p < min_mask:
                        eq = cv.cvtColor(processed, cv.COLOR_BGR2GRAY)
                        eq = cv.equalizeHist(eq)
                        eq = cv.cvtColor(eq, cv.COLOR_GRAY2BGR)
                        mask = prp.extract_cnn_mask(eq, net)
                        white_p = prp.calc_white_percentage(mask)
                        method = "Mask R-CNN + histequalize"

                    if mask is not None and white_p > min_mask:
                        break;

                    sharpening = True
                    if n == 1:
                        double_size = False

                if mask is None or white_p < min_mask:
                    print("[INFO] All methods failed")
                    mask = None
                    if writelog:
                        log_file.write("Oliva {0} FALLITO\n".format(i+1))
                else:
                    # RIPORTA LE DIMENSIONI DELLA MASCHERA A QUELLE DELLA IMMAGINE ORIGINALE
                    if double_size:
                        nw = roi_bgr_padding.shape[1]
                        nh = roi_bgr_padding.shape[0]
                        dim = (nw, nh)
                        mask = cv.resize(mask, dim, interpolation=cv.INTER_LINEAR)
                    if visualize:
                        cv.imshow("mask", mask)
                        cv.waitKey(0)
                        cv.destroyAllWindows()
                    """ CODICE PER GENERARE LE IMMAGINI PER RESNET, CREARE PRIMA LE MATURAZIONI E LE CARTELLE
                    msc = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
                    masked = cv.bitwise_and(roi_bgr_padding, msc)
                    cv.imwrite("mask5/{0}/{1}.jpg".format(y3[successes + success], successes + success), masked)
                    cv.imwrite("mask3/{0}/{1}.jpg".format(y4[successes + success], successes + success), masked)
                    """
                    success = success + 1
                    if writelog:
                        log_file.write("Oliva {0} SUCCESSO,\tPIXEL MASCHERA = {1:.2f}%, sharpening = {2}, double size "
                                       "= {3}, metodo = {4}\n".format(i+1, white_p, sharpening, double_size, method))

                    if hsv:
                        stack = calc_histogram(roi_hsv_padding, bins, "HSV", mask, plot)
                    else:
                        stack = calc_histogram(roi_bgr_padding, bins, "BGR", mask, plot)
            else:
                if hsv:
                    stack = calc_histogram(roi_hsv, bins, "HSV", plot=plot)
                else:
                    stack = calc_histogram(roi_bgr, bins, "BGR", plot=plot)

            hists.append(stack)  # ISTOGRAMMI CALCOLATI PER TEST IMMEDIATO

            if writedataset:
                write_dataset(mask, roi_bgr, roi_hsv, roi_bgr_padding, roi_hsv_padding)

            i = i + 1
        except Exception:
            i = i + 1
            continue

    if writelog:
        log_file.write("RILEVAMENTI: {0} SU {1}, TEMPO: {2}\n".format(success, i, time.time() - start_time))
        log_file.write("----------------------\n\n")
        log_file.close()

    hists_conv = np.array(hists)  # Converte la lista in un array numpy
    return hists_conv, success, i


def write_dataset(mask, roi_bgr, roi_hsv, roi_bgr_extra, roi_hsv_extra):
    """
    Funzione automatica di scrittura del dataset, eseguita all'interno di extract_histograms

    :param mask: La maschera da passare alla funzione di calcolo istogramma
    :param roi_bgr: Immagine dell'oliva in formato BGR (DEFAULT OPENCV) senza padding
    :param roi_hsv: Immagine dell'oliva in formato HSV senza padding
    :param roi_bgr_extra: Immagine dell'oliva in formato BGR con padding per il calcolo con maschera
    :param roi_hsv_extra: Immagine dell'oliva in formato HSV con padding per il calcolo con maschera
    """
    if mask is not None:
        for g in range(3, 6):
            bins = pow(2, g)
            file = open("rgb_" + str(bins) + "bin_masked.txt", 'a')
            rgb_full = calc_histogram(roi_bgr_extra, bins)
            hsv_full = calc_histogram(roi_hsv_extra, bins)
            for f in rgb_full:
                file.write(str(int(f)) + " ")
            file.write("\n")
            file.close()
            file = open("hsv_" + str(bins) + "bin_masked.txt", 'a')
            for f in hsv_full:
                file.write(str(int(f)) + " ")
            file.write("\n")
            file.close()

    for g in range(3, 6):
        bins = pow(2, g)
        file = open("rgb_" + str(bins) + "bin.txt", 'a')
        rgb_full = calc_histogram(roi_bgr, bins)
        hsv_full = calc_histogram(roi_hsv, bins)
        for f in rgb_full:
            file.write(str(int(f)) + " ")
        file.write("\n")
        file.close()
        file = open("hsv_" + str(bins) + "bin.txt", 'a')
        for f in hsv_full:
            file.write(str(int(f)) + " ")
        file.write("\n")
        file.close()


def calc_histogram(image, bins=256, label="123", mask=None, plot=False):
    """
    Calcola e traccia l'istogramma per I 3 canali

    :param image: L'immagine di input, deve essere a 3 canali
    :param bins: Il numero di intervalli su cui calcolare l'istogramma, possibilimente in potenze di 2
    :param label: Stringa di almeno 3 caratteri per etichettare i 3 canali nel grafico
    :param mask: La maschera per calcolare l'istogramma solamente in un area specifica
    :param plot: Per scegliere di visualizzare l'istogramma
    :return: Array numpy formato dai valori calcolato per i 3 canali (dimensione 3 * bins)
    """
    assert (len(image.shape) == 3 and image.shape[2] == 3), "The image does not have 3 channels"
    if mask is None:
        mask = np.full(image.shape[0:2], 255, np.uint8)

    c1, c2, c3 = cv.split(image)

    r1 = cv.calcHist([c1], [0], mask, [bins], [0, 256])  # Istogramma di opencv è 10x più veloce di numpy
    # axis.hist(c1.ravel(), bins=256, range=[0, 256], label=label[0]) # Hist di matplotlib ricalcola le frequenze
    r2 = cv.calcHist([c2], [0], mask, [bins], [0, 256])
    r3 = cv.calcHist([c3], [0], mask, [bins], [0, 256])

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.bar(np.arange(bins), r1.ravel(), color='tab:blue')
        ax1.set_title(label[0])
        ax1.label_outer()
        ax2.bar(np.arange(bins), r2.ravel(), color='tab:green')
        ax2.set_title(label[1])
        ax2.label_outer()
        ax3.bar(np.arange(bins), r3.ravel(), color='tab:red')
        ax3.set_title(label[2])
        ax3.label_outer()
        plt.show()

    stacked = np.vstack((r1, r2, r3))  # AFFIANCA I 3 ARRAY DI VALORI IN UNICO NDARRAY
    return stacked


def test_classifiers(x_train, x_test, y_train, y_test, bins, mask, colorspace, classes):
    """
    Funzione per testare i sei modelli di classificazione di Sklearn, calcola inoltre l'F1_SCORE e le matrici di
    confusione

    :param x_train: L'array input di allenamento
    :param x_test: L'array input di test
    :param y_train: Le label corrette per l'allenamento
    :param y_test: Le label corrette per il test
    :param bins: Numero di bins per salvare il file dei punteggi
    :param mask: Per indicare se sono state applicate le maschere, per il file dei punteggi
    :param colorspace: Per indicare lo spazio di colori del dataset, per il file dei punteggi
    :param classes: Per indicare il numero di classi, per il file dei punteggi
    """
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        tree.DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=200),
        MLPClassifier(alpha=1, max_iter=2000),
        AdaBoostClassifier()]
    names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost"]
    for name, clf in zip(names, classifiers):
        clf = clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        weighted = f1_score(y_test, y_pred, average="weighted")
        with open("scores/f1_weighted.csv", "a") as f:
            txt = "{0}, {1}, {2}, {3}, {4}, {5}\n".format(name, bins, mask, colorspace, classes, weighted)
            f.write(txt)
        none = f1_score(y_test, y_pred, average=None)
        vals = ""
        for n in none:
            vals = vals + str(n) + ", "
        with open("scores/f1_none.csv", "a") as f:
            txt = "{0}, {1}, {2}, {3}, {4}, {5}\n".format(name, bins, mask, colorspace, classes, vals)
            f.write(txt)

        if mask:
            mstring = "_mask"
        else:
            mstring = ""

        disp = plot_confusion_matrix(clf, x_test, y_test, normalize="true")
        disp.ax_.set_title(name)
        print(disp.confusion_matrix)
        plt.savefig("confusion_matrixes/{0}_{1}{2}_{3}classes_{4}.png".format(bins, colorspace,mstring, classes, name))
        # plt.show()
        plt.close('all')


def calc_f1_score(dataset):
    """
    Funzione per calcolare rapidamente tutti i punteggi F1 e le matrici di confusione

    :param dataset: 1 o 2 per scegliere rispettivamente i due dataset presenti, oppure 'both' per mischiarli
    """
    mask = True
    for u in range(2):
        cs = "hsv"
        for i in range(2):
            three_c = False
            for k in range(2):
                for j in range(3, 6):
                    bins = pow(2, j)
                    x1, y1 = utils.load_training_data(bins, cs, masked=mask, dataset=dataset, three_classes=three_c)
                    print(x1.shape[0])
                    x_train, x_test, y_train, y_test = utils.split_data(x1, y1)
                    print("[INFO] testing...")
                    if three_c:
                        c = "3"
                    else:
                        c = "5"
                    test_classifiers(x_train, x_test, y_train, y_test, bins, mask, cs, c)
                three_c = True
            cs = "rgb"
        mask = False


successes = 0
total = 0
for k in range(1):
    _, s, t = extract_histograms("images/{0}.jpg".format(k+1), "labels/{0}.txt".format(k+1), min_mask=20)
    successes = successes + s
    total = total + t

percent = (successes / total) * 100
result = open("result.txt", 'a')
result.write(str(successes)+" SU "+str(total)+" SUCCESSI, {:.2f}".format(percent)+"%\n")
result.close()


