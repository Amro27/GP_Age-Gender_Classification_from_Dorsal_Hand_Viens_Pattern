import numpy as np
from scipy.misc import toimage
import os, os.path
import scipy.misc
import cv2
import csv
from skimage import exposure
from skimage import feature
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics.cluster import entropy
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.svm import SVC

# Local Ternary Pattern 3x3
from sklearn.ensemble import VotingClassifier

import numpy as np

from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import QObject, pyqtSignal
import cv2
# import pyqtgraph as pg
# import pyqtgraph
import random
import sys, time
import pandas as pd

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

Ui_MainWindow, QMainWindow = loadUiType('Grad_Pro_GUI.ui')


class XStream(QObject):
    _stdout = None
    _stderr = None

    messageWritten = pyqtSignal(str)

    def flush(self):
        pass

    def fileno(self):
        return -1

    def write(self, msg):
        if (not self.signalsBlocked()):
            self.messageWritten.emit(unicode(msg))

    @staticmethod
    def stdout():
        if (not XStream._stdout):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if (not XStream._stderr):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr


def LocalTernaryPattern(img, t):
    numrows = len(img) - 2  # number of rows
    numcols = len(img[0]) - 2  # number of columns
    x = np.zeros((numrows, numcols))  # [[0 for i in range(numrows-1)] for j in range(numcols)]
    bi = []
    for i in range(1, numrows - 1):
        for j in range(1, numcols - 1):
            if img[i - 1][j - 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i][j - 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i + 1][j - 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i + 1][j] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i + 1][j + 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i][j + 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i - 1][j + 1] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)
            if img[i - 1][j] > img[i][j]:
                bi.append(1)
            else:
                bi.append(0)

            dec = int(''.join(str(x) for x in bi), base=2)
            bi = [0]
            x[i - 1][j - 1] = dec

    # x=x*((1/x.max())*9)#normalization

    numrows = len(img) - 2
    numcols = len(img[0]) - 2
    y = np.zeros((numrows, numcols))
    bi = []
    d = t
    for i in range(1, numrows - 1):
        for j in range(1, numcols - 1):
            if (img[i - 1][j - 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i][j - 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i + 1][j - 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i + 1][j] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i + 1][j + 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i][j + 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i - 1][j + 1] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)
            if (img[i - 1][j] - img[i][j]) >= d:
                bi.append(1)
            else:
                bi.append(0)

            dec = int(''.join(str(x) for x in bi), base=2)
            bi = [0]
            y[i - 1][j - 1] = dec
    # y=y*((1/y.max())*9)
    z = x + y
    z = z * ((1 / y.max()) * 9)
    return z


def region_of_interest(image_list=[], image_listROI=[]):
    global minr
    for im in image_list:
        # apply threshold
        thresh = threshold_otsu(im)
        bw = closing(im > thresh, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=im)
        minr = 0
        minc = 0
        maxr = 0
        maxc = 0
        r1 = 0
        r2 = 0
        c1 = 0
        c2 = 0
        for region in regionprops(label_image):
            if region.area >= 100:
                minr, minc, maxr, maxc = region.bbox
        r1 = minr + 75
        r2 = maxr - 75
        c1 = minc + 75
        c2 = maxc - 75
        if r1 > r2:
            # temp1=r1
            r1 = 77
            r2 = 340
        if c1 > c2:
            # temp2=c1
            c1 = 150
            c2 = 500
        # print r1, r2, c1, c2
        crop = im[r1:r2, c1:c2]
        resized_image = cv2.resize(crop, (200, 200))
        image_listROI.append(resized_image)


def morphology_filter(image_list=[], image_listmor=[]):
    for im in image_list:
        kernel = np.ones((3, 3), np.uint8)
        img = im
        image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        image_listmor.append(image)


def bilateral_filter(image_list=[], image_listBL=[]):
    for im in image_list:
        image = cv2.bilateralFilter(im, 9, 75, 75)
        image_listBL.append(image)


def median_filter(image_list=[], image_listmB=[]):
    for im in image_list:
        image = cv2.medianBlur(im, 5)
        image_listmB.append(image)


# (Contrast Limited Adaptive Histogram Equalization)
def CLAHE(image_list=[], image_listcl=[]):
    for im in image_list:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(im)
        image_listcl.append(cl1)


# threshold
def image_preprocessing(image_listcl=[], image_preprocessing=[]):
    for images in image_listcl:
        img = scipy.misc.imresize(images, 0.5)
        h, w = img.shape
        ret, img = cv2.threshold(img, 15, 255, cv2.THRESH_TOZERO)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 9)
        new_img = th2

        for i in range(0, h - 5, 5):
            for j in range(0, w - 5, 5):
                for x in range(i, i + 7):
                    for y in range(j, j + 7):
                        if (th2[x, y]) == 0:
                            for m in range(i - 3, i + 3):
                                for n in range(j - 3, j + 3):
                                    new_img[m, n] = 0
        image_preprocessing.append(new_img)


def Apply_comatrix(image_list=[], new_list=[]):
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    for im in image_list:
        glcm = greycomatrix(im, distances=distances, angles=angles, symmetric=True, normed=True)
        new_list.append(glcm)


def Apply_LTP(image_preprocessing=[], LTP=[]):
    for images in image_preprocessing:
        y = LocalTernaryPattern(images, 5)
        LTP.append(y)


def Apply_LBP(image_preprocessing=[], LBP=[]):
    for images in image_preprocessing:
        y = local_binary_pattern(images, 8, 1, method='uniform')
        LBP.append(y)


def create_csv_file_con(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['contrast']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            for i in range(len(hist1)):
                hist1[i] = float(hist1[i] - min(hist1)) / (max(hist1) - min(hist1))
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_dis(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['dissimilarity']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            for i in range(len(hist1)):
                hist1[i] = float(hist1[i] - min(hist1)) / (max(hist1) - min(hist1))
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_cor(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['correlation']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            for i in range(len(hist1)):
                hist1[i] = float(hist1[i] - min(hist1)) / (max(hist1) - min(hist1))
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_ASM(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['ASM']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            for i in range(len(hist1)):
                hist1[i] = float(hist1[i] - min(hist1)) / (max(hist1) - min(hist1))
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_con_Avg(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['contrast']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_dis_Avg(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['dissimilarity']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_cor_Avg(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['correlation']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_ASM_Avg(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['ASM']
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1 = np.hstack([greycoprops(list1[i], prop).ravel() for prop in properties])
            f1 = []
            # f11 = (np.sum(list(hist1), axis=0))/(len(list1))
            # f1.append(f11)
            m = image_ID[0]
            f1.append(m)
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1.extend(list(hist1))
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


            # feature vectors from histogram


def create_csv_file(file_name, x, image_ID=[], list1=[]):
    with open(file_name, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for i in range(len(list1)):
            hist1, b1 = exposure.histogram(list1[i], nbins=8)
            f1 = []
            m = image_ID[0]
            f1.append(m)
            f1.extend(list(hist1))
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        # pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)

        XStream.stdout().messageWritten.connect(self.textBrowser.insertPlainText)
        XStream.stdout().messageWritten.connect(self.textBrowser.ensureCursorVisible)
        XStream.stderr().messageWritten.connect(self.textBrowser.insertPlainText)
        XStream.stderr().messageWritten.connect(self.textBrowser.ensureCursorVisible)

        self.Capture_btn.clicked.connect(self.cap_image)
        self.All_btn.clicked.connect(self.Do_It_All)
        self.Run_btn.clicked.connect(self.run)
        self.Classify_btn.clicked.connect(self.classify)
        self.SVM_btn.setChecked(True)
        self.KNN_btn.setChecked(True)
        self.LBP_btn.setChecked(True)
        self.LTP_btn.setChecked(True)
        self.ASM_btn.setChecked(True)
        self.Contrast_btn.setChecked(True)
        self.Correlation_btn.setChecked(True)
        self.Dissimilarity_btn.setChecked(True)
        self.GENDER_btn.setChecked(True)
        self.AGE_btn.setChecked(True)
        self.Comatrix_btn.setChecked(True)
        self.LBP_LTP_btn.setChecked(True)
        self.Comatrix_LBP_LTP_btn.setChecked(True)
        self.Binary_btn.setChecked(True)
        self.Grayscale_btn.setChecked(True)

        self.centralwidget.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 20, 120, 250);}\n"))
        self.All_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 250, 0, 40);}\n"))
        self.Run_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 250, 0, 0);}\n"))
        self.textBrowser.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 250, 250, 250);}\n"))
        self.Capture_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 0, 250, 0);}\n"))
        self.Classify_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 40, 190, 255);}\n"))

    def run(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.progressBar.setValue(self.completed)

        img = cv2.imread('cap4.jpg', 0)
        image_listM = []
        image_listM.append(img)
        image_IDM = [1]

        image_listMROI = []
        region_of_interest(image_listM, image_listMROI)

        image_listMmB = []
        morphology_filter(image_listMROI, image_listMmB)

        image_listMmor = []
        bilateral_filter(image_listMmB, image_listMmor)

        image_listMBL = []
        median_filter(image_listMmor, image_listMBL)

        if self.Grayscale_btn.isChecked() == True:

            # image_preprocessingM = []
            # image_preprocessing(image_listMcl, image_preprocessingM)

            image_preprocessingM = []
            CLAHE(image_listMBL, image_preprocessingM)

            TRY_comatM = []
            Apply_comatrix(image_preprocessingM, TRY_comatM)

            LBPM = []
            Apply_LBP(image_preprocessingM, LBPM)

            LTPM = []
            Apply_LTP(image_preprocessingM, LTPM)

            create_csv_file_con("Test_contrast_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis("Test_dissimilarity_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor("Test_correlation_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM("Test_ASM_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_con_Avg("Test_contrast_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis_Avg("Test_dissimilarity_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor_Avg("Test_correlation_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM_Avg("Test_ASM_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file("Test_LTP_G.csv", 1, image_IDM, LTPM)
            create_csv_file("Test_LBP_G.csv", 1, image_IDM, LBPM)

        elif self.Binary_btn.isChecked() == True:

            image_listMcl = []
            CLAHE(image_listMBL, image_listMcl)

            image_preprocessingM = []
            image_preprocessing(image_listMcl, image_preprocessingM)

            TRY_comatM = []
            Apply_comatrix(image_preprocessingM, TRY_comatM)

            LBPM = []
            Apply_LBP(image_preprocessingM, LBPM)

            LTPM = []
            Apply_LTP(image_preprocessingM, LTPM)

            create_csv_file_con("Test_contrast.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis("Test_dissimilarity.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor("Test_correlation.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM("Test_ASM.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_con_Avg("Test_contrast_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis_Avg("Test_dissimilarity_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor_Avg("Test_correlation_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM_Avg("Test_ASM_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file("Test_LTP.csv", 1, image_IDM, LTPM)
            create_csv_file("Test_LBP.csv", 1, image_IDM, LBPM)

    def classify(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.progressBar.setValue(self.completed)
        global Xs, Ys, X3, knn, X_trainA, y_trainA, X_testA, X_trainG, y_trainG, X_testG, XYCor, yYCor, XTCor, yMCor, XMCor, XFCor, yFCor, XOCor, yOCor, yFCon, XFCon, XMCon, yMCon, yYCon, XYCon, XOCon, yOCon, XTCon, yMDis, XMDis, XFDis, yFDis, XTDis, yYDis, XYDis, XODis, yODis, yFASM, yMASM, XMASM, XFASM, XTASM, yYASM, XYASM, XOASM, yOASM, yMT, XMT, XFT, yFT, XTT, yYT, XYT, XOT, yOT, yML, XML, XFL, yFL, XTL, yYL, XYL, XOL, yOL
        if self.Binary_btn.isChecked() == True:
            names = ['class']
            dfYL = pd.read_csv('Young_LBP.csv', header=None, names=names)
            djYL = pd.read_csv('Young_LBP.csv', header=None)
            jYL = djYL.values
            kYL = np.delete(jYL, 0, 1)
            XYL = np.delete(kYL, 8, 1)
            yYL = np.array(dfYL['class'])

            dfOL = pd.read_csv('Old_LBP.csv', header=None, names=names)
            djOL = pd.read_csv('Old_LBP.csv', header=None)
            jOL = djOL.values
            kOL = np.delete(jOL, 0, 1)
            XOL = np.delete(kOL, 8, 1)
            yOL = np.array(dfOL['class'])

            dfML = pd.read_csv('Male_LBP.csv', header=None, names=names)
            djML = pd.read_csv('Male_LBP.csv', header=None)
            jML = djML.values
            kML = np.delete(jML, 0, 1)
            XML = np.delete(kML, 8, 1)
            yML = np.array(dfML['class'])

            dfFL = pd.read_csv('Female_LBP.csv', header=None, names=names)
            djFL = pd.read_csv('Female_LBP.csv', header=None)
            jFL = djFL.values
            kFL = np.delete(jFL, 0, 1)
            XFL = np.delete(kFL, 8, 1)
            yFL = np.array(dfFL['class'])

            names = ['class']
            dfYT = pd.read_csv('Young_LTP.csv', header=None, names=names)
            djYT = pd.read_csv('Young_LTP.csv', header=None)
            jYT = djYT.values
            kYT = np.delete(jYT, 0, 1)
            XYT = np.delete(kYT, 8, 1)
            yYT = np.array(dfYT['class'])

            dfOT = pd.read_csv('Old_LTP.csv', header=None, names=names)
            djOT = pd.read_csv('Old_LTP.csv', header=None)
            jOT = djOT.values
            kOT = np.delete(jOT, 0, 1)
            XOT = np.delete(kOT, 8, 1)
            yOT = np.array(dfOT['class'])

            dfMT = pd.read_csv('Male_LTP.csv', header=None, names=names)
            djMT = pd.read_csv('Male_LTP.csv', header=None)
            jMT = djMT.values
            kMT = np.delete(jMT, 0, 1)
            XMT = np.delete(kMT, 8, 1)
            yMT = np.array(dfMT['class'])

            dfFT = pd.read_csv('Female_LTP.csv', header=None, names=names)
            djFT = pd.read_csv('Female_LTP.csv', header=None)
            jFT = djFT.values
            kFT = np.delete(jFT, 0, 1)
            XFT = np.delete(kFT, 8, 1)
            yFT = np.array(dfFT['class'])

            names = ['class']
            dfYASM = pd.read_csv('Young_ASM.csv', header=None, names=names)
            djYASM = pd.read_csv('Young_ASM.csv', header=None)
            jYASM = djYASM.values
            kYASM = np.delete(jYASM, 0, 1)
            XYASM = np.delete(kYASM, 12, 1)
            yYASM = np.array(dfYASM['class'])

            dfOASM = pd.read_csv('Old_ASM.csv', header=None, names=names)
            djOASM = pd.read_csv('Old_ASM.csv', header=None)
            jOASM = djOASM.values
            kOASM = np.delete(jOASM, 0, 1)
            XOASM = np.delete(kOASM, 12, 1)
            yOASM = np.array(dfOASM['class'])

            dfMASM = pd.read_csv('Male_ASM.csv', header=None, names=names)
            djMASM = pd.read_csv('Male_ASM.csv', header=None)
            jMASM = djMASM.values
            kMASM = np.delete(jMASM, 0, 1)
            XMASM = np.delete(kMASM, 12, 1)
            yMASM = np.array(dfMASM['class'])

            dfFASM = pd.read_csv('Female_ASM.csv', header=None, names=names)
            djFASM = pd.read_csv('Female_ASM.csv', header=None)
            jFASM = djFASM.values
            kFASM = np.delete(jFASM, 0, 1)
            XFASM = np.delete(kFASM, 12, 1)
            yFASM = np.array(dfFASM['class'])

            names = ['class']
            dfYDis = pd.read_csv('Young_dissimilarity.csv', header=None, names=names)
            djYDis = pd.read_csv('Young_dissimilarity.csv', header=None)
            jYDis = djYDis.values
            kYDis = np.delete(jYDis, 0, 1)
            XYDis = np.delete(kYDis, 12, 1)
            yYDis = np.array(dfYDis['class'])

            dfODis = pd.read_csv('Old_dissimilarity.csv', header=None, names=names)
            djODis = pd.read_csv('Old_dissimilarity.csv', header=None)
            jODis = djODis.values
            kODis = np.delete(jODis, 0, 1)
            XODis = np.delete(kODis, 12, 1)
            yODis = np.array(dfODis['class'])

            dfMDis = pd.read_csv('Male_dissimilarity.csv', header=None, names=names)
            djMDis = pd.read_csv('Male_dissimilarity.csv', header=None)
            jMDis = djMDis.values
            kMDis = np.delete(jMDis, 0, 1)
            XMDis = np.delete(kMDis, 12, 1)
            yMDis = np.array(dfMDis['class'])

            dfFDis = pd.read_csv('Female_dissimilarity.csv', header=None, names=names)
            djFDis = pd.read_csv('Female_dissimilarity.csv', header=None)
            jFDis = djFDis.values
            kFDis = np.delete(jFDis, 0, 1)
            XFDis = np.delete(kFDis, 12, 1)
            yFDis = np.array(dfFDis['class'])

            names = ['class']
            dfYCon = pd.read_csv('Young_contrast.csv', header=None, names=names)
            djYCon = pd.read_csv('Young_contrast.csv', header=None)
            jYCon = djYCon.values
            kYCon = np.delete(jYCon, 0, 1)
            XYCon = np.delete(kYCon, 12, 1)
            yYCon = np.array(dfYCon['class'])

            dfOCon = pd.read_csv('Old_contrast.csv', header=None, names=names)
            djOCon = pd.read_csv('Old_contrast.csv', header=None)
            jOCon = djOCon.values
            kOCon = np.delete(jOCon, 0, 1)
            XOCon = np.delete(kOCon, 12, 1)
            yOCon = np.array(dfOCon['class'])

            dfMCon = pd.read_csv('Male_contrast.csv', header=None, names=names)
            djMCon = pd.read_csv('Male_contrast.csv', header=None)
            jMCon = djMCon.values
            kMCon = np.delete(jMCon, 0, 1)
            XMCon = np.delete(kMCon, 12, 1)
            yMCon = np.array(dfMCon['class'])

            dfFCon = pd.read_csv('Female_contrast.csv', header=None, names=names)
            djFCon = pd.read_csv('Female_contrast.csv', header=None)
            jFCon = djFCon.values
            kFCon = np.delete(jFCon, 0, 1)
            XFCon = np.delete(kFCon, 12, 1)
            yFCon = np.array(dfFCon['class'])

            names = ['class']
            dfYCor = pd.read_csv('Young_correlation.csv', header=None, names=names)
            djYCor = pd.read_csv('Young_correlation.csv', header=None)
            jYCor = djYCor.values
            kYCor = np.delete(jYCor, 0, 1)
            XYCor = np.delete(kYCor, 12, 1)
            yYCor = np.array(dfYCor['class'])

            dfOCor = pd.read_csv('Old_correlation.csv', header=None, names=names)
            djOCor = pd.read_csv('Old_correlation.csv', header=None)
            jOCor = djOCor.values
            kOCor = np.delete(jOCor, 0, 1)
            XOCor = np.delete(kOCor, 12, 1)
            yOCor = np.array(dfOCor['class'])

            dfMCor = pd.read_csv('Male_correlation.csv', header=None, names=names)
            djMCor = pd.read_csv('Male_correlation.csv', header=None)
            jMCor = djMCor.values
            kMCor = np.delete(jMCor, 0, 1)
            XMCor = np.delete(kMCor, 12, 1)
            yMCor = np.array(dfMCor['class'])

            dfFCor = pd.read_csv('Female_correlation.csv', header=None, names=names)
            djFCor = pd.read_csv('Female_correlation.csv', header=None)
            jFCor = djFCor.values
            kFCor = np.delete(jFCor, 0, 1)
            XFCor = np.delete(kFCor, 12, 1)
            yFCor = np.array(dfFCor['class'])

            djTL = pd.read_csv('Test_LBP.csv', header=None)
            jTL = djTL.values
            kTL = np.delete(jTL, 0, 1)
            XTL = np.delete(kTL, 8, 1)

            djTT = pd.read_csv('Test_LTP.csv', header=None)
            jTT = djTT.values
            kTT = np.delete(jTT, 0, 1)
            XTT = np.delete(kTT, 8, 1)

            djTASM = pd.read_csv('Test_ASM.csv', header=None)
            jTASM = djTASM.values
            kTASM = np.delete(jTASM, 0, 1)
            XTASM = np.delete(kTASM, 12, 1)

            djTDis = pd.read_csv('Test_dissimilarity.csv', header=None)
            jTDis = djTDis.values
            kTDis = np.delete(jTDis, 0, 1)
            XTDis = np.delete(kTDis, 12, 1)

            djTCor = pd.read_csv('Test_correlation.csv', header=None)
            jTCor = djTCor.values
            kTCor = np.delete(jTCor, 0, 1)
            XTCor = np.delete(kTCor, 12, 1)

            djTCon = pd.read_csv('Test_contrast.csv', header=None)
            jTCon = djTCon.values
            kTCon = np.delete(jTCon, 0, 1)
            XTCon = np.delete(kTCon, 12, 1)

        if self.Grayscale_btn.isChecked() == True:
            names = ['class']
            dfYL = pd.read_csv('Young_LBP_G.csv', header=None, names=names)
            djYL = pd.read_csv('Young_LBP_G.csv', header=None)
            jYL = djYL.values
            kYL = np.delete(jYL, 0, 1)
            XYL = np.delete(kYL, 8, 1)
            yYL = np.array(dfYL['class'])

            dfOL = pd.read_csv('Old_LBP_G.csv', header=None, names=names)
            djOL = pd.read_csv('Old_LBP_G.csv', header=None)
            jOL = djOL.values
            kOL = np.delete(jOL, 0, 1)
            XOL = np.delete(kOL, 8, 1)
            yOL = np.array(dfOL['class'])

            dfML = pd.read_csv('Male_LBP_G.csv', header=None, names=names)
            djML = pd.read_csv('Male_LBP_G.csv', header=None)
            jML = djML.values
            kML = np.delete(jML, 0, 1)
            XML = np.delete(kML, 8, 1)
            yML = np.array(dfML['class'])

            dfFL = pd.read_csv('Female_LBP_G.csv', header=None, names=names)
            djFL = pd.read_csv('Female_LBP_G.csv', header=None)
            jFL = djFL.values
            kFL = np.delete(jFL, 0, 1)
            XFL = np.delete(kFL, 8, 1)
            yFL = np.array(dfFL['class'])

            names = ['class']
            dfYT = pd.read_csv('Young_LTP_G.csv', header=None, names=names)
            djYT = pd.read_csv('Young_LTP_G.csv', header=None)
            jYT = djYT.values
            kYT = np.delete(jYT, 0, 1)
            XYT = np.delete(kYT, 8, 1)
            yYT = np.array(dfYT['class'])

            dfOT = pd.read_csv('Old_LTP_G.csv', header=None, names=names)
            djOT = pd.read_csv('Old_LTP_G.csv', header=None)
            jOT = djOT.values
            kOT = np.delete(jOT, 0, 1)
            XOT = np.delete(kOT, 8, 1)
            yOT = np.array(dfOT['class'])

            dfMT = pd.read_csv('Male_LTP_G.csv', header=None, names=names)
            djMT = pd.read_csv('Male_LTP_G.csv', header=None)
            jMT = djMT.values
            kMT = np.delete(jMT, 0, 1)
            XMT = np.delete(kMT, 8, 1)
            yMT = np.array(dfMT['class'])

            dfFT = pd.read_csv('Female_LTP_G.csv', header=None, names=names)
            djFT = pd.read_csv('Female_LTP_G.csv', header=None)
            jFT = djFT.values
            kFT = np.delete(jFT, 0, 1)
            XFT = np.delete(kFT, 8, 1)
            yFT = np.array(dfFT['class'])

            names = ['class']
            dfYASM = pd.read_csv('Young_ASM_G.csv', header=None, names=names)
            djYASM = pd.read_csv('Young_ASM_G.csv', header=None)
            jYASM = djYASM.values
            kYASM = np.delete(jYASM, 0, 1)
            XYASM = np.delete(kYASM, 12, 1)
            yYASM = np.array(dfYASM['class'])

            dfOASM = pd.read_csv('Old_ASM_G.csv', header=None, names=names)
            djOASM = pd.read_csv('Old_ASM_G.csv', header=None)
            jOASM = djOASM.values
            kOASM = np.delete(jOASM, 0, 1)
            XOASM = np.delete(kOASM, 12, 1)
            yOASM = np.array(dfOASM['class'])

            dfMASM = pd.read_csv('Male_ASM_G.csv', header=None, names=names)
            djMASM = pd.read_csv('Male_ASM_G.csv', header=None)
            jMASM = djMASM.values
            kMASM = np.delete(jMASM, 0, 1)
            XMASM = np.delete(kMASM, 12, 1)
            yMASM = np.array(dfMASM['class'])

            dfFASM = pd.read_csv('Female_ASM_G.csv', header=None, names=names)
            djFASM = pd.read_csv('Female_ASM_G.csv', header=None)
            jFASM = djFASM.values
            kFASM = np.delete(jFASM, 0, 1)
            XFASM = np.delete(kFASM, 12, 1)
            yFASM = np.array(dfFASM['class'])

            names = ['class']
            dfYDis = pd.read_csv('Young_dissimilarity_G.csv', header=None, names=names)
            djYDis = pd.read_csv('Young_dissimilarity_G.csv', header=None)
            jYDis = djYDis.values
            kYDis = np.delete(jYDis, 0, 1)
            XYDis = np.delete(kYDis, 12, 1)
            yYDis = np.array(dfYDis['class'])

            dfODis = pd.read_csv('Old_dissimilarity_G.csv', header=None, names=names)
            djODis = pd.read_csv('Old_dissimilarity_G.csv', header=None)
            jODis = djODis.values
            kODis = np.delete(jODis, 0, 1)
            XODis = np.delete(kODis, 12, 1)
            yODis = np.array(dfODis['class'])

            dfMDis = pd.read_csv('Male_dissimilarity_G.csv', header=None, names=names)
            djMDis = pd.read_csv('Male_dissimilarity_G.csv', header=None)
            jMDis = djMDis.values
            kMDis = np.delete(jMDis, 0, 1)
            XMDis = np.delete(kMDis, 12, 1)
            yMDis = np.array(dfMDis['class'])

            dfFDis = pd.read_csv('Female_dissimilarity_G.csv', header=None, names=names)
            djFDis = pd.read_csv('Female_dissimilarity_G.csv', header=None)
            jFDis = djFDis.values
            kFDis = np.delete(jFDis, 0, 1)
            XFDis = np.delete(kFDis, 12, 1)
            yFDis = np.array(dfFDis['class'])

            names = ['class']
            dfYCon = pd.read_csv('Young_contrast_G.csv', header=None, names=names)
            djYCon = pd.read_csv('Young_contrast_G.csv', header=None)
            jYCon = djYCon.values
            kYCon = np.delete(jYCon, 0, 1)
            XYCon = np.delete(kYCon, 12, 1)
            yYCon = np.array(dfYCon['class'])

            dfOCon = pd.read_csv('Old_contrast_G.csv', header=None, names=names)
            djOCon = pd.read_csv('Old_contrast_G.csv', header=None)
            jOCon = djOCon.values
            kOCon = np.delete(jOCon, 0, 1)
            XOCon = np.delete(kOCon, 12, 1)
            yOCon = np.array(dfOCon['class'])

            dfMCon = pd.read_csv('Male_contrast_G.csv', header=None, names=names)
            djMCon = pd.read_csv('Male_contrast_G.csv', header=None)
            jMCon = djMCon.values
            kMCon = np.delete(jMCon, 0, 1)
            XMCon = np.delete(kMCon, 12, 1)
            yMCon = np.array(dfMCon['class'])

            dfFCon = pd.read_csv('Female_contrast_G.csv', header=None, names=names)
            djFCon = pd.read_csv('Female_contrast_G.csv', header=None)
            jFCon = djFCon.values
            kFCon = np.delete(jFCon, 0, 1)
            XFCon = np.delete(kFCon, 12, 1)
            yFCon = np.array(dfFCon['class'])

            names = ['class']
            dfYCor = pd.read_csv('Young_correlation_G.csv', header=None, names=names)
            djYCor = pd.read_csv('Young_correlation_G.csv', header=None)
            jYCor = djYCor.values
            kYCor = np.delete(jYCor, 0, 1)
            XYCor = np.delete(kYCor, 12, 1)
            yYCor = np.array(dfYCor['class'])

            dfOCor = pd.read_csv('Old_correlation_G.csv', header=None, names=names)
            djOCor = pd.read_csv('Old_correlation_G.csv', header=None)
            jOCor = djOCor.values
            kOCor = np.delete(jOCor, 0, 1)
            XOCor = np.delete(kOCor, 12, 1)
            yOCor = np.array(dfOCor['class'])

            dfMCor = pd.read_csv('Male_correlation_G.csv', header=None, names=names)
            djMCor = pd.read_csv('Male_correlation_G.csv', header=None)
            jMCor = djMCor.values
            kMCor = np.delete(jMCor, 0, 1)
            XMCor = np.delete(kMCor, 12, 1)
            yMCor = np.array(dfMCor['class'])

            dfFCor = pd.read_csv('Female_correlation_G.csv', header=None, names=names)
            djFCor = pd.read_csv('Female_correlation_G.csv', header=None)
            jFCor = djFCor.values
            kFCor = np.delete(jFCor, 0, 1)
            XFCor = np.delete(kFCor, 12, 1)
            yFCor = np.array(dfFCor['class'])

            djTL = pd.read_csv('Test_LBP_G.csv', header=None)
            jTL = djTL.values
            kTL = np.delete(jTL, 0, 1)
            XTL = np.delete(kTL, 8, 1)

            djTT = pd.read_csv('Test_LTP_G.csv', header=None)
            jTT = djTT.values
            kTT = np.delete(jTT, 0, 1)
            XTT = np.delete(kTT, 8, 1)

            djTASM = pd.read_csv('Test_ASM_G.csv', header=None)
            jTASM = djTASM.values
            kTASM = np.delete(jTASM, 0, 1)
            XTASM = np.delete(kTASM, 12, 1)

            djTDis = pd.read_csv('Test_dissimilarity_G.csv', header=None)
            jTDis = djTDis.values
            kTDis = np.delete(jTDis, 0, 1)
            XTDis = np.delete(kTDis, 12, 1)

            djTCor = pd.read_csv('Test_correlation_G.csv', header=None)
            jTCor = djTCor.values
            kTCor = np.delete(jTCor, 0, 1)
            XTCor = np.delete(kTCor, 12, 1)

            djTCon = pd.read_csv('Test_contrast_G.csv', header=None)
            jTCon = djTCon.values
            kTCon = np.delete(jTCon, 0, 1)
            XTCon = np.delete(kTCon, 12, 1)

        if self.LBP_btn.isChecked() == True:
            X_trainA = np.concatenate((XYL, XOL), axis=0)
            y_trainA = np.concatenate((yYL, yOL), axis=0)
            X_testA = XTL

            X_trainG = np.concatenate((XML, XFL), axis=0)
            y_trainG = np.concatenate((yML, yFL), axis=0)
            X_testG = XTL

        if self.LTP_btn.isChecked() == True:
            X_trainA = np.concatenate((XYT, XOT), axis=0)
            y_trainA = np.concatenate((yYT, yOT), axis=0)
            X_testA = XTT

            X_trainG = np.concatenate((XMT, XFT), axis=0)
            y_trainG = np.concatenate((yMT, yFT), axis=0)
            X_testG = XTT

        if self.ASM_btn.isChecked() == True:
            X_trainA = np.concatenate((XYASM, XOASM), axis=0)
            y_trainA = np.concatenate((yYASM, yOASM), axis=0)
            X_testA = XTASM

            X_trainG = np.concatenate((XMASM, XFASM), axis=0)
            y_trainG = np.concatenate((yMASM, yFASM), axis=0)
            X_testG = XTASM

        if self.Dissimilarity_btn.isChecked() == True:
            X_trainA = np.concatenate((XYDis, XODis), axis=0)
            y_trainA = np.concatenate((yYDis, yODis), axis=0)
            X_testA = XTDis

            X_trainG = np.concatenate((XMDis, XFDis), axis=0)
            y_trainG = np.concatenate((yMDis, yFDis), axis=0)
            X_testG = XTDis

        if self.Contrast_btn.isChecked() == True:
            X_trainA = np.concatenate((XYCon, XOCon), axis=0)
            y_trainA = np.concatenate((yYCon, yOCon), axis=0)
            X_testA = XTCon

            X_trainG = np.concatenate((XMCon, XFCon), axis=0)
            y_trainG = np.concatenate((yMCon, yFCon), axis=0)
            X_testG = XTCon

        if self.Correlation_btn.isChecked() == True:
            X_trainA = np.concatenate((XYCor, XOCor), axis=0)
            y_trainA = np.concatenate((yYCor, yOCor), axis=0)
            X_testA = XTCor

            X_trainG = np.concatenate((XMCor, XFCor), axis=0)
            y_trainG = np.concatenate((yMCor, yFCor), axis=0)
            X_testG = XTCor

        if self.Comatrix_btn.isChecked() == True:
            Xy_train = np.concatenate((XYASM, XYCon, XYCor, XYDis), axis=1)
            Xo_train = np.concatenate((XOASM, XOCon, XOCor, XODis), axis=1)
            Xm_train = np.concatenate((XMASM, XMCon, XMCor, XMDis), axis=1)
            Xf_train = np.concatenate((XFASM, XFCon, XFCor, XFDis), axis=1)

            Xt = np.concatenate((XTASM, XTCon, XTCor, XTDis), axis=1)

            X_trainA = np.concatenate((Xy_train, Xo_train), axis=0)
            y_trainA = np.concatenate((yYCor, yOCor), axis=0)
            X_testA = Xt

            X_trainG = np.concatenate((Xm_train, Xf_train), axis=0)
            y_trainG = np.concatenate((yMCor, yFCor), axis=0)
            X_testG = Xt

            # print len(Xt),len(X_trainA),len(y_trainA)

        if self.LBP_LTP_btn.isChecked() == True:
            Xy_train = np.concatenate((XYL, XYT), axis=1)
            Xo_train = np.concatenate((XOL, XOT), axis=1)
            Xm_train = np.concatenate((XML, XMT), axis=1)
            Xf_train = np.concatenate((XFL, XFT), axis=1)

            Xt = np.concatenate((XTL, XTT), axis=1)

            X_trainA = np.concatenate((Xy_train, Xo_train), axis=0)
            y_trainA = np.concatenate((yYCor, yOCor), axis=0)
            X_testA = Xt

            X_trainG = np.concatenate((Xm_train, Xf_train), axis=0)
            y_trainG = np.concatenate((yMCor, yFCor), axis=0)
            X_testG = Xt

        if self.Comatrix_LBP_LTP_btn.isChecked() == True:
            Xy_train = np.concatenate((XYASM, XYCon, XYCor, XYDis, XYL, XYT), axis=1)
            Xo_train = np.concatenate((XOASM, XOCon, XOCor, XODis, XOL, XOT), axis=1)
            Xm_train = np.concatenate((XMASM, XMCon, XMCor, XMDis, XML, XMT), axis=1)
            Xf_train = np.concatenate((XFASM, XFCon, XFCor, XFDis, XFL, XFT), axis=1)

            Xt = np.concatenate((XTASM, XTCon, XTCor, XTDis, XTL, XTT), axis=1)

            X_trainA = np.concatenate((Xy_train, Xo_train), axis=0)
            y_trainA = np.concatenate((yYCor, yOCor), axis=0)
            X_testA = Xt

            X_trainG = np.concatenate((Xm_train, Xf_train), axis=0)
            y_trainG = np.concatenate((yMCor, yFCor), axis=0)
            X_testG = Xt

        if self.SVM_btn.isChecked() == True:
            knn = svm.SVC(kernel='linear', gamma=1)
        elif self.KNN_btn.isChecked() == True:
            knn = KNeighborsClassifier()
        elif self.KNN_SVM_btn.isChecked() == True:
            clf1 = svm.SVC(kernel='linear', gamma=1)
            clf2 = KNeighborsClassifier()
            knn = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2)], voting='hard')
        if self.AGE_btn.isChecked() == True:
            Xs = X_trainA
            Ys = y_trainA
            X3 = X_testA
        elif self.GENDER_btn.isChecked() == True:
            Xs = X_trainG
            Ys = y_trainG
            X3 = X_testG
        knn.fit(Xs, Ys)
        pred = knn.predict(X3)
        if self.AGE_btn.isChecked() == True:
            #print pred
            if pred == 1:
                self.Classify_result.setText("Old.")
            else:
                self.Classify_result.setText("Young.")
        elif self.GENDER_btn.isChecked() == True:
            if pred == 1:
                self.Classify_result.setText("Male.")
            else:
                self.Classify_result.setText("Female.")

    def cap_image(self):
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                out = cv2.imwrite('cap4.jpg', frame)
                break
        cap.release()
        cv2.destroyAllWindows()
        # self.Image_label.setPixmap(QtGui.QPixmap(_fromUtf8("cap4.jpg")))

        return 'cap4.jpg'

    def Do_It_All(self):
        self.All_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 250, 0, 0);}\n"))
        cap = cv2.VideoCapture(2)
        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                out = cv2.imwrite('cap5.jpg', frame)
                break
        cap.release()
        cv2.destroyAllWindows()

        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.progressBar.setValue(self.completed)

        img = cv2.imread('cap5.jpg', 0)
        image_listM = []
        image_listM.append(img)
        image_IDM = [1]

        image_listMROI = []
        region_of_interest(image_listM, image_listMROI)

        image_listMmB = []
        morphology_filter(image_listMROI, image_listMmB)

        image_listMmor = []
        bilateral_filter(image_listMmB, image_listMmor)

        image_listMBL = []
        median_filter(image_listMmor, image_listMBL)

        if self.Grayscale_btn.isChecked() == True:

            # image_preprocessingM = []
            # image_preprocessing(image_listMcl, image_preprocessingM)

            image_preprocessingM = []
            CLAHE(image_listMBL, image_preprocessingM)

            TRY_comatM = []
            Apply_comatrix(image_preprocessingM, TRY_comatM)

            LBPM = []
            Apply_LBP(image_preprocessingM, LBPM)

            LTPM = []
            Apply_LTP(image_preprocessingM, LTPM)

            create_csv_file_con("Test_contrast_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis("Test_dissimilarity_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor("Test_correlation_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM("Test_ASM_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_con_Avg("Test_contrast_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis_Avg("Test_dissimilarity_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor_Avg("Test_correlation_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM_Avg("Test_ASM_Avg_G.csv", 1, image_IDM, TRY_comatM)
            create_csv_file("Test_LTP_G.csv", 1, image_IDM, LTPM)
            create_csv_file("Test_LBP_G.csv", 1, image_IDM, LBPM)

        elif self.Binary_btn.isChecked() == True:

            image_listMcl = []
            CLAHE(image_listMBL, image_listMcl)

            image_preprocessingM = []
            image_preprocessing(image_listMcl, image_preprocessingM)

            TRY_comatM = []
            Apply_comatrix(image_preprocessingM, TRY_comatM)

            LBPM = []
            Apply_LBP(image_preprocessingM, LBPM)

            LTPM = []
            Apply_LTP(image_preprocessingM, LTPM)

            create_csv_file_con("Test_contrast.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis("Test_dissimilarity.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor("Test_correlation.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM("Test_ASM.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_con_Avg("Test_contrast_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_dis_Avg("Test_dissimilarity_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_cor_Avg("Test_correlation_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file_ASM_Avg("Test_ASM_Avg.csv", 1, image_IDM, TRY_comatM)
            create_csv_file("Test_LTP.csv", 1, image_IDM, LTPM)
            create_csv_file("Test_LBP.csv", 1, image_IDM, LBPM)

        global Xs, Ys, X3, knn, X_trainA, y_trainA, X_testA, X_trainG, y_trainG, X_testG, XYCor, yYCor, XTCor, yMCor, XMCor, XFCor, yFCor, XOCor, yOCor, yFCon, XFCon, XMCon, yMCon, yYCon, XYCon, XOCon, yOCon, XTCon, yMDis, XMDis, XFDis, yFDis, XTDis, yYDis, XYDis, XODis, yODis, yFASM, yMASM, XMASM, XFASM, XTASM, yYASM, XYASM, XOASM, yOASM, yMT, XMT, XFT, yFT, XTT, yYT, XYT, XOT, yOT, yML, XML, XFL, yFL, XTL, yYL, XYL, XOL, yOL
        if self.Binary_btn.isChecked() == True:
            names = ['class']
            dfYL = pd.read_csv('Young_LBP.csv', header=None, names=names)
            djYL = pd.read_csv('Young_LBP.csv', header=None)
            jYL = djYL.values
            kYL = np.delete(jYL, 0, 1)
            XYL = np.delete(kYL, 8, 1)
            yYL = np.array(dfYL['class'])

            dfOL = pd.read_csv('Old_LBP.csv', header=None, names=names)
            djOL = pd.read_csv('Old_LBP.csv', header=None)
            jOL = djOL.values
            kOL = np.delete(jOL, 0, 1)
            XOL = np.delete(kOL, 8, 1)
            yOL = np.array(dfOL['class'])

            dfML = pd.read_csv('Male_LBP.csv', header=None, names=names)
            djML = pd.read_csv('Male_LBP.csv', header=None)
            jML = djML.values
            kML = np.delete(jML, 0, 1)
            XML = np.delete(kML, 8, 1)
            yML = np.array(dfML['class'])

            dfFL = pd.read_csv('Female_LBP.csv', header=None, names=names)
            djFL = pd.read_csv('Female_LBP.csv', header=None)
            jFL = djFL.values
            kFL = np.delete(jFL, 0, 1)
            XFL = np.delete(kFL, 8, 1)
            yFL = np.array(dfFL['class'])

            names = ['class']
            dfYT = pd.read_csv('Young_LTP.csv', header=None, names=names)
            djYT = pd.read_csv('Young_LTP.csv', header=None)
            jYT = djYT.values
            kYT = np.delete(jYT, 0, 1)
            XYT = np.delete(kYT, 8, 1)
            yYT = np.array(dfYT['class'])

            dfOT = pd.read_csv('Old_LTP.csv', header=None, names=names)
            djOT = pd.read_csv('Old_LTP.csv', header=None)
            jOT = djOT.values
            kOT = np.delete(jOT, 0, 1)
            XOT = np.delete(kOT, 8, 1)
            yOT = np.array(dfOT['class'])

            dfMT = pd.read_csv('Male_LTP.csv', header=None, names=names)
            djMT = pd.read_csv('Male_LTP.csv', header=None)
            jMT = djMT.values
            kMT = np.delete(jMT, 0, 1)
            XMT = np.delete(kMT, 8, 1)
            yMT = np.array(dfMT['class'])

            dfFT = pd.read_csv('Female_LTP.csv', header=None, names=names)
            djFT = pd.read_csv('Female_LTP.csv', header=None)
            jFT = djFT.values
            kFT = np.delete(jFT, 0, 1)
            XFT = np.delete(kFT, 8, 1)
            yFT = np.array(dfFT['class'])

            names = ['class']
            dfYASM = pd.read_csv('Young_ASM.csv', header=None, names=names)
            djYASM = pd.read_csv('Young_ASM.csv', header=None)
            jYASM = djYASM.values
            kYASM = np.delete(jYASM, 0, 1)
            XYASM = np.delete(kYASM, 12, 1)
            yYASM = np.array(dfYASM['class'])

            dfOASM = pd.read_csv('Old_ASM.csv', header=None, names=names)
            djOASM = pd.read_csv('Old_ASM.csv', header=None)
            jOASM = djOASM.values
            kOASM = np.delete(jOASM, 0, 1)
            XOASM = np.delete(kOASM, 12, 1)
            yOASM = np.array(dfOASM['class'])

            dfMASM = pd.read_csv('Male_ASM.csv', header=None, names=names)
            djMASM = pd.read_csv('Male_ASM.csv', header=None)
            jMASM = djMASM.values
            kMASM = np.delete(jMASM, 0, 1)
            XMASM = np.delete(kMASM, 12, 1)
            yMASM = np.array(dfMASM['class'])

            dfFASM = pd.read_csv('Female_ASM.csv', header=None, names=names)
            djFASM = pd.read_csv('Female_ASM.csv', header=None)
            jFASM = djFASM.values
            kFASM = np.delete(jFASM, 0, 1)
            XFASM = np.delete(kFASM, 12, 1)
            yFASM = np.array(dfFASM['class'])

            names = ['class']
            dfYDis = pd.read_csv('Young_dissimilarity.csv', header=None, names=names)
            djYDis = pd.read_csv('Young_dissimilarity.csv', header=None)
            jYDis = djYDis.values
            kYDis = np.delete(jYDis, 0, 1)
            XYDis = np.delete(kYDis, 12, 1)
            yYDis = np.array(dfYDis['class'])

            dfODis = pd.read_csv('Old_dissimilarity.csv', header=None, names=names)
            djODis = pd.read_csv('Old_dissimilarity.csv', header=None)
            jODis = djODis.values
            kODis = np.delete(jODis, 0, 1)
            XODis = np.delete(kODis, 12, 1)
            yODis = np.array(dfODis['class'])

            dfMDis = pd.read_csv('Male_dissimilarity.csv', header=None, names=names)
            djMDis = pd.read_csv('Male_dissimilarity.csv', header=None)
            jMDis = djMDis.values
            kMDis = np.delete(jMDis, 0, 1)
            XMDis = np.delete(kMDis, 12, 1)
            yMDis = np.array(dfMDis['class'])

            dfFDis = pd.read_csv('Female_dissimilarity.csv', header=None, names=names)
            djFDis = pd.read_csv('Female_dissimilarity.csv', header=None)
            jFDis = djFDis.values
            kFDis = np.delete(jFDis, 0, 1)
            XFDis = np.delete(kFDis, 12, 1)
            yFDis = np.array(dfFDis['class'])

            names = ['class']
            dfYCon = pd.read_csv('Young_contrast.csv', header=None, names=names)
            djYCon = pd.read_csv('Young_contrast.csv', header=None)
            jYCon = djYCon.values
            kYCon = np.delete(jYCon, 0, 1)
            XYCon = np.delete(kYCon, 12, 1)
            yYCon = np.array(dfYCon['class'])

            dfOCon = pd.read_csv('Old_contrast.csv', header=None, names=names)
            djOCon = pd.read_csv('Old_contrast.csv', header=None)
            jOCon = djOCon.values
            kOCon = np.delete(jOCon, 0, 1)
            XOCon = np.delete(kOCon, 12, 1)
            yOCon = np.array(dfOCon['class'])

            dfMCon = pd.read_csv('Male_contrast.csv', header=None, names=names)
            djMCon = pd.read_csv('Male_contrast.csv', header=None)
            jMCon = djMCon.values
            kMCon = np.delete(jMCon, 0, 1)
            XMCon = np.delete(kMCon, 12, 1)
            yMCon = np.array(dfMCon['class'])

            dfFCon = pd.read_csv('Female_contrast.csv', header=None, names=names)
            djFCon = pd.read_csv('Female_contrast.csv', header=None)
            jFCon = djFCon.values
            kFCon = np.delete(jFCon, 0, 1)
            XFCon = np.delete(kFCon, 12, 1)
            yFCon = np.array(dfFCon['class'])

            names = ['class']
            dfYCor = pd.read_csv('Young_correlation.csv', header=None, names=names)
            djYCor = pd.read_csv('Young_correlation.csv', header=None)
            jYCor = djYCor.values
            kYCor = np.delete(jYCor, 0, 1)
            XYCor = np.delete(kYCor, 12, 1)
            yYCor = np.array(dfYCor['class'])

            dfOCor = pd.read_csv('Old_correlation.csv', header=None, names=names)
            djOCor = pd.read_csv('Old_correlation.csv', header=None)
            jOCor = djOCor.values
            kOCor = np.delete(jOCor, 0, 1)
            XOCor = np.delete(kOCor, 12, 1)
            yOCor = np.array(dfOCor['class'])

            dfMCor = pd.read_csv('Male_correlation.csv', header=None, names=names)
            djMCor = pd.read_csv('Male_correlation.csv', header=None)
            jMCor = djMCor.values
            kMCor = np.delete(jMCor, 0, 1)
            XMCor = np.delete(kMCor, 12, 1)
            yMCor = np.array(dfMCor['class'])

            dfFCor = pd.read_csv('Female_correlation.csv', header=None, names=names)
            djFCor = pd.read_csv('Female_correlation.csv', header=None)
            jFCor = djFCor.values
            kFCor = np.delete(jFCor, 0, 1)
            XFCor = np.delete(kFCor, 12, 1)
            yFCor = np.array(dfFCor['class'])

            djTL = pd.read_csv('Test_LBP.csv', header=None)
            jTL = djTL.values
            kTL = np.delete(jTL, 0, 1)
            XTL = np.delete(kTL, 8, 1)

            djTT = pd.read_csv('Test_LTP.csv', header=None)
            jTT = djTT.values
            kTT = np.delete(jTT, 0, 1)
            XTT = np.delete(kTT, 8, 1)

            djTASM = pd.read_csv('Test_ASM.csv', header=None)
            jTASM = djTASM.values
            kTASM = np.delete(jTASM, 0, 1)
            XTASM = np.delete(kTASM, 12, 1)

            djTDis = pd.read_csv('Test_dissimilarity.csv', header=None)
            jTDis = djTDis.values
            kTDis = np.delete(jTDis, 0, 1)
            XTDis = np.delete(kTDis, 12, 1)

            djTCor = pd.read_csv('Test_correlation.csv', header=None)
            jTCor = djTCor.values
            kTCor = np.delete(jTCor, 0, 1)
            XTCor = np.delete(kTCor, 12, 1)

            djTCon = pd.read_csv('Test_contrast.csv', header=None)
            jTCon = djTCon.values
            kTCon = np.delete(jTCon, 0, 1)
            XTCon = np.delete(kTCon, 12, 1)

        if self.Grayscale_btn.isChecked() == True:
            names = ['class']
            dfYL = pd.read_csv('Young_LBP_G.csv', header=None, names=names)
            djYL = pd.read_csv('Young_LBP_G.csv', header=None)
            jYL = djYL.values
            kYL = np.delete(jYL, 0, 1)
            XYL = np.delete(kYL, 8, 1)
            yYL = np.array(dfYL['class'])

            dfOL = pd.read_csv('Old_LBP_G.csv', header=None, names=names)
            djOL = pd.read_csv('Old_LBP_G.csv', header=None)
            jOL = djOL.values
            kOL = np.delete(jOL, 0, 1)
            XOL = np.delete(kOL, 8, 1)
            yOL = np.array(dfOL['class'])

            dfML = pd.read_csv('Male_LBP_G.csv', header=None, names=names)
            djML = pd.read_csv('Male_LBP_G.csv', header=None)
            jML = djML.values
            kML = np.delete(jML, 0, 1)
            XML = np.delete(kML, 8, 1)
            yML = np.array(dfML['class'])

            dfFL = pd.read_csv('Female_LBP_G.csv', header=None, names=names)
            djFL = pd.read_csv('Female_LBP_G.csv', header=None)
            jFL = djFL.values
            kFL = np.delete(jFL, 0, 1)
            XFL = np.delete(kFL, 8, 1)
            yFL = np.array(dfFL['class'])

            names = ['class']
            dfYT = pd.read_csv('Young_LTP_G.csv', header=None, names=names)
            djYT = pd.read_csv('Young_LTP_G.csv', header=None)
            jYT = djYT.values
            kYT = np.delete(jYT, 0, 1)
            XYT = np.delete(kYT, 8, 1)
            yYT = np.array(dfYT['class'])

            dfOT = pd.read_csv('Old_LTP_G.csv', header=None, names=names)
            djOT = pd.read_csv('Old_LTP_G.csv', header=None)
            jOT = djOT.values
            kOT = np.delete(jOT, 0, 1)
            XOT = np.delete(kOT, 8, 1)
            yOT = np.array(dfOT['class'])

            dfMT = pd.read_csv('Male_LTP_G.csv', header=None, names=names)
            djMT = pd.read_csv('Male_LTP_G.csv', header=None)
            jMT = djMT.values
            kMT = np.delete(jMT, 0, 1)
            XMT = np.delete(kMT, 8, 1)
            yMT = np.array(dfMT['class'])

            dfFT = pd.read_csv('Female_LTP_G.csv', header=None, names=names)
            djFT = pd.read_csv('Female_LTP_G.csv', header=None)
            jFT = djFT.values
            kFT = np.delete(jFT, 0, 1)
            XFT = np.delete(kFT, 8, 1)
            yFT = np.array(dfFT['class'])

            names = ['class']
            dfYASM = pd.read_csv('Young_ASM_G.csv', header=None, names=names)
            djYASM = pd.read_csv('Young_ASM_G.csv', header=None)
            jYASM = djYASM.values
            kYASM = np.delete(jYASM, 0, 1)
            XYASM = np.delete(kYASM, 12, 1)
            yYASM = np.array(dfYASM['class'])

            dfOASM = pd.read_csv('Old_ASM_G.csv', header=None, names=names)
            djOASM = pd.read_csv('Old_ASM_G.csv', header=None)
            jOASM = djOASM.values
            kOASM = np.delete(jOASM, 0, 1)
            XOASM = np.delete(kOASM, 12, 1)
            yOASM = np.array(dfOASM['class'])

            dfMASM = pd.read_csv('Male_ASM_G.csv', header=None, names=names)
            djMASM = pd.read_csv('Male_ASM_G.csv', header=None)
            jMASM = djMASM.values
            kMASM = np.delete(jMASM, 0, 1)
            XMASM = np.delete(kMASM, 12, 1)
            yMASM = np.array(dfMASM['class'])

            dfFASM = pd.read_csv('Female_ASM_G.csv', header=None, names=names)
            djFASM = pd.read_csv('Female_ASM_G.csv', header=None)
            jFASM = djFASM.values
            kFASM = np.delete(jFASM, 0, 1)
            XFASM = np.delete(kFASM, 12, 1)
            yFASM = np.array(dfFASM['class'])

            names = ['class']
            dfYDis = pd.read_csv('Young_dissimilarity_G.csv', header=None, names=names)
            djYDis = pd.read_csv('Young_dissimilarity_G.csv', header=None)
            jYDis = djYDis.values
            kYDis = np.delete(jYDis, 0, 1)
            XYDis = np.delete(kYDis, 12, 1)
            yYDis = np.array(dfYDis['class'])

            dfODis = pd.read_csv('Old_dissimilarity_G.csv', header=None, names=names)
            djODis = pd.read_csv('Old_dissimilarity_G.csv', header=None)
            jODis = djODis.values
            kODis = np.delete(jODis, 0, 1)
            XODis = np.delete(kODis, 12, 1)
            yODis = np.array(dfODis['class'])

            dfMDis = pd.read_csv('Male_dissimilarity_G.csv', header=None, names=names)
            djMDis = pd.read_csv('Male_dissimilarity_G.csv', header=None)
            jMDis = djMDis.values
            kMDis = np.delete(jMDis, 0, 1)
            XMDis = np.delete(kMDis, 12, 1)
            yMDis = np.array(dfMDis['class'])

            dfFDis = pd.read_csv('Female_dissimilarity_G.csv', header=None, names=names)
            djFDis = pd.read_csv('Female_dissimilarity_G.csv', header=None)
            jFDis = djFDis.values
            kFDis = np.delete(jFDis, 0, 1)
            XFDis = np.delete(kFDis, 12, 1)
            yFDis = np.array(dfFDis['class'])

            names = ['class']
            dfYCon = pd.read_csv('Young_contrast_G.csv', header=None, names=names)
            djYCon = pd.read_csv('Young_contrast_G.csv', header=None)
            jYCon = djYCon.values
            kYCon = np.delete(jYCon, 0, 1)
            XYCon = np.delete(kYCon, 12, 1)
            yYCon = np.array(dfYCon['class'])

            dfOCon = pd.read_csv('Old_contrast_G.csv', header=None, names=names)
            djOCon = pd.read_csv('Old_contrast_G.csv', header=None)
            jOCon = djOCon.values
            kOCon = np.delete(jOCon, 0, 1)
            XOCon = np.delete(kOCon, 12, 1)
            yOCon = np.array(dfOCon['class'])

            dfMCon = pd.read_csv('Male_contrast_G.csv', header=None, names=names)
            djMCon = pd.read_csv('Male_contrast_G.csv', header=None)
            jMCon = djMCon.values
            kMCon = np.delete(jMCon, 0, 1)
            XMCon = np.delete(kMCon, 12, 1)
            yMCon = np.array(dfMCon['class'])

            dfFCon = pd.read_csv('Female_contrast_G.csv', header=None, names=names)
            djFCon = pd.read_csv('Female_contrast_G.csv', header=None)
            jFCon = djFCon.values
            kFCon = np.delete(jFCon, 0, 1)
            XFCon = np.delete(kFCon, 12, 1)
            yFCon = np.array(dfFCon['class'])

            names = ['class']
            dfYCor = pd.read_csv('Young_correlation_G.csv', header=None, names=names)
            djYCor = pd.read_csv('Young_correlation_G.csv', header=None)
            jYCor = djYCor.values
            kYCor = np.delete(jYCor, 0, 1)
            XYCor = np.delete(kYCor, 12, 1)
            yYCor = np.array(dfYCor['class'])

            dfOCor = pd.read_csv('Old_correlation_G.csv', header=None, names=names)
            djOCor = pd.read_csv('Old_correlation_G.csv', header=None)
            jOCor = djOCor.values
            kOCor = np.delete(jOCor, 0, 1)
            XOCor = np.delete(kOCor, 12, 1)
            yOCor = np.array(dfOCor['class'])

            dfMCor = pd.read_csv('Male_correlation_G.csv', header=None, names=names)
            djMCor = pd.read_csv('Male_correlation_G.csv', header=None)
            jMCor = djMCor.values
            kMCor = np.delete(jMCor, 0, 1)
            XMCor = np.delete(kMCor, 12, 1)
            yMCor = np.array(dfMCor['class'])

            dfFCor = pd.read_csv('Female_correlation_G.csv', header=None, names=names)
            djFCor = pd.read_csv('Female_correlation_G.csv', header=None)
            jFCor = djFCor.values
            kFCor = np.delete(jFCor, 0, 1)
            XFCor = np.delete(kFCor, 12, 1)
            yFCor = np.array(dfFCor['class'])

            djTL = pd.read_csv('Test_LBP_G.csv', header=None)
            jTL = djTL.values
            kTL = np.delete(jTL, 0, 1)
            XTL = np.delete(kTL, 8, 1)

            djTT = pd.read_csv('Test_LTP_G.csv', header=None)
            jTT = djTT.values
            kTT = np.delete(jTT, 0, 1)
            XTT = np.delete(kTT, 8, 1)

            djTASM = pd.read_csv('Test_ASM_G.csv', header=None)
            jTASM = djTASM.values
            kTASM = np.delete(jTASM, 0, 1)
            XTASM = np.delete(kTASM, 12, 1)

            djTDis = pd.read_csv('Test_dissimilarity_G.csv', header=None)
            jTDis = djTDis.values
            kTDis = np.delete(jTDis, 0, 1)
            XTDis = np.delete(kTDis, 12, 1)

            djTCor = pd.read_csv('Test_correlation_G.csv', header=None)
            jTCor = djTCor.values
            kTCor = np.delete(jTCor, 0, 1)
            XTCor = np.delete(kTCor, 12, 1)

            djTCon = pd.read_csv('Test_contrast_G.csv', header=None)
            jTCon = djTCon.values
            kTCon = np.delete(jTCon, 0, 1)
            XTCon = np.delete(kTCon, 12, 1)

        Xy_train1 = np.concatenate((XYASM, XYCon, XYCor, XYDis), axis=1)
        Xo_train1 = np.concatenate((XOASM, XOCon, XOCor, XODis), axis=1)
        Xm_train1 = np.concatenate((XMASM, XMCon, XMCor, XMDis), axis=1)
        Xf_train1 = np.concatenate((XFASM, XFCon, XFCor, XFDis), axis=1)

        Xt1 = np.concatenate((XTASM, XTCon, XTCor, XTDis), axis=1)

        X_trainA1 = np.concatenate((Xy_train1, Xo_train1), axis=0)
        y_trainA1 = np.concatenate((yYCor, yOCor), axis=0)
        X_testA1 = Xt1

        X_trainG1 = np.concatenate((Xm_train1, Xf_train1), axis=0)
        y_trainG1 = np.concatenate((yMCor, yFCor), axis=0)
        X_testG1 = Xt1

        # print len(Xt),len(X_trainA),len(y_trainA)


        Xy_train2 = XYL
        Xo_train2 = XOL
        Xm_train2 = XML
        Xf_train2 = XFL

        Xt2 = XTL


        #Xy_train2 = np.concatenate((XYL, XYT), axis=1)
        #Xo_train2 = np.concatenate((XOL, XOT), axis=1)
        #Xm_train2 = np.concatenate((XML, XMT), axis=1)
        #Xf_train2 = np.concatenate((XFL, XFT), axis=1)

        #Xt2 = np.concatenate((XTL, XTT), axis=1)

        X_trainA2 = np.concatenate((Xy_train2, Xo_train2), axis=0)
        y_trainA2 = np.concatenate((yYL, yOL), axis=0)
        X_testA2 = Xt2

        X_trainG2 = np.concatenate((Xm_train2, Xf_train2), axis=0)
        y_trainG2 = np.concatenate((yML, yFL), axis=0)
        X_testG2 = Xt2

        Xy_train3 = XYT
        Xo_train3 = XOT
        Xm_train3 = XMT
        Xf_train3 = XFT

        Xt3 = XTT


        X_trainA3 = np.concatenate((Xy_train3, Xo_train3), axis=0)
        y_trainA3 = np.concatenate((yYT, yOT), axis=0)
        X_testA3 = Xt3

        X_trainG3 = np.concatenate((Xm_train3, Xf_train3), axis=0)
        y_trainG3 = np.concatenate((yMT, yFT), axis=0)
        X_testG3 = Xt3

        #Xy_train3 = np.concatenate((XYASM, XYCon, XYCor, XYDis, XYL, XYT), axis=1)
        #Xo_train3 = np.concatenate((XOASM, XOCon, XOCor, XODis, XOL, XOT), axis=1)
        #Xm_train3 = np.concatenate((XMASM, XMCon, XMCor, XMDis, XML, XMT), axis=1)
        #Xf_train3 = np.concatenate((XFASM, XFCon, XFCor, XFDis, XFL, XFT), axis=1)

        #Xt3 = np.concatenate((XTASM, XTCon, XTCor, XTDis, XTL, XTT), axis=1)

        #X_trainA3 = np.concatenate((Xy_train3, Xo_train3), axis=0)
        #y_trainA3 = np.concatenate((yYCor, yOCor), axis=0)
        #X_testA3 = Xt3

        #X_trainG3 = np.concatenate((Xm_train3, Xf_train3), axis=0)
        #y_trainG3 = np.concatenate((yMCor, yFCor), axis=0)
        #X_testG3 = Xt3

        clf1 = svm.SVC(kernel='linear', gamma=1)
        clf2 = KNeighborsClassifier()
        knn = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2)], voting='hard')
        XsA1 = X_trainA1
        YsA1 = y_trainA1
        X3A1 = X_testA1
        XsA2 = X_trainA2
        YsA2 = y_trainA2
        X3A2 = X_testA2
        XsA3 = X_trainA3
        YsA3 = y_trainA3
        X3A3 = X_testA3

        XsG1 = X_trainG1
        YsG1 = y_trainG1
        X3G1 = X_testG1
        XsG2 = X_trainG2
        YsG2 = y_trainG2
        X3G2 = X_testG2
        XsG3 = X_trainG3
        YsG3 = y_trainG3
        X3G3 = X_testG3

        self.completed = 0

        while self.completed < 100:
            self.completed += 0.0001
            self.progressBar.setValue(self.completed)

        knn.fit(XsA1, YsA1)
        pred1 = knn.predict(X3A1)
        knn.fit(XsA2, YsA2)
        pred2 = knn.predict(X3A2)
        knn.fit(XsA3, YsA3)
        pred3 = knn.predict(X3A3)
        predA = 0
        if pred1 == 1:
            predA += 1
        if pred2 == 1:
            predA += 1
        if pred3 == 1:
            predA += 1
        #predA = pred1 + pred2 + pred3

        knn.fit(XsG1, YsG1)
        pred4 = knn.predict(X3G1)
        knn.fit(XsG2, YsG2)
        pred5 = knn.predict(X3G2)
        knn.fit(XsG3, YsG3)
        pred6 = knn.predict(X3G3)
        predG=0
        if pred4==1:
            predG+=1
        if pred5==1:
            predG+=1
        if pred6==1:
            predG+=1
        #predG = pred4 + pred5 + pred6

        #print predG

        if (predA >= 2) & (predG >= 2):
            self.Classify_result.setText("Old & Male")
        elif (predA >= 2) & (predG < 2 ):
            self.Classify_result.setText("Old & Female")
        elif (predA < 2) & (predG < 2 ):
            self.Classify_result.setText("Young & Female")
        elif (predA < 2) & (predG >= 2) :
            self.Classify_result.setText("Young & Male")
        self.All_btn.setStyleSheet(_fromUtf8("QWidget{background-color: rgb( 0, 250, 0);}\n"))

    try:
        _fromUtf8 = QtCore.QString.fromUtf8
    except AttributeError:
        def _fromUtf8(s):
            return s

    try:
        _encoding = QtGui.QApplication.UnicodeUTF8

        def _translate(context, text, disambig):
            return QtGui.QApplication.translate(context, text, disambig, _encoding)
    except AttributeError:
        def _translate(context, text, disambig):
            return QtGui.QApplication.translate(context, text, disambig)



            # self.pushButton_20.clicked.connect( self.start_thread0 )
            # self.pushButton_20.setStyleSheet( "background-color: green" ) ## This is to change Button Color



            # def start_thread0(self):##Function of any Button
            #    self.listen.EMG = np.empty( [0, 8] )
            #    threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
            #    self.flag_thread0 = True
            #    self.thread0 = threading.Thread(target = self.loop0)
            #    self.thread0.start()


if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
