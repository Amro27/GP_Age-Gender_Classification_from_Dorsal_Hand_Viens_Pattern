from pylab import *
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
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.svm import SVC



# Local Ternary Pattern 3x3
from sklearn.ensemble import VotingClassifier


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
    for im in image_list:
        # apply threshold
        thresh = threshold_otsu(im)
        bw = closing(im > thresh, square(3))

        # remove artifacts connected to image border
        cleared = clear_border(bw)

        # label image regions
        label_image = label(cleared)
        image_label_overlay = label2rgb(label_image, image=im)

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


# Adaptive threshold
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
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
            m = image_ID[i]
            f1.append(m)
            f1.extend(list(hist1))
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


img = cv2.imread('try.jpg',0)
image_listM = []
image_listM.append(img)
image_IDM=[1]


image_listMROI=[]
region_of_interest(image_listM, image_listMROI)

image_listMmB=[]
morphology_filter(image_listMROI, image_listMmB)

image_listMmor=[]
bilateral_filter(image_listMmB, image_listMmor)

image_listMBL=[]
median_filter(image_listMmor, image_listMBL)

image_listMcl=[]
CLAHE(image_listMBL, image_listMcl)

image_preprocessingM=[]
image_preprocessing(image_listMcl, image_preprocessingM)

TRY_comatM=[]
Apply_comatrix(image_preprocessingM, TRY_comatM)

LBPM=[]
Apply_LBP(image_preprocessingM, LBPM)

LTPM=[]
Apply_LTP(image_preprocessingM, LTPM)

create_csv_file_con("Test_contrast.csv",None,image_IDM,TRY_comatM)
create_csv_file_dis("Test_dissimilarity.csv",None,image_IDM,TRY_comatM)
create_csv_file_cor("Test_correlation.csv",None,image_IDM,TRY_comatM)
create_csv_file_ASM("Test_ASM.csv",None,image_IDM,TRY_comatM)
create_csv_file_con_Avg("Test_contrast_Avg.csv",None,image_IDM,TRY_comatM)
create_csv_file_dis_Avg("Test_dissimilarity_Avg.csv",None,image_IDM,TRY_comatM)
create_csv_file_cor_Avg("Test_correlation_Avg.csv",None,image_IDM,TRY_comatM)
create_csv_file_ASM_Avg("Test_ASM_Avg.csv",None,image_IDM,TRY_comatM)
create_csv_file("Test_LTP.csv",None,image_IDM,LTPM)
create_csv_file("Test_LBP.csv",None,image_IDM,LBPM)



names = ['class']
df1 = pd.read_csv('Young_LBP.csv', header=None, names=names)
dj1 = pd.read_csv('Young_LBP.csv', header=None)
j1 = dj1.values
k1 = np.delete(j1, 0, 1)
X1 = np.delete(k1, 8, 1)
y1 = np.array(df1['class'])

df2 = pd.read_csv('Old_LBP.csv', header=None, names=names)
dj2 = pd.read_csv('Old_LBP.csv', header=None)
j2 = dj2.values
k2 = np.delete(j2, 0, 1)
X2 = np.delete(k2, 8, 1)
y2 = np.array(df2['class'])


names = ['class']
dfx = pd.read_csv('Male_LBP.csv', header=None, names=names)
djx = pd.read_csv('Male_LBP.csv', header=None)
jx = djx.values
kx = np.delete(jx, 0, 1)
Xx = np.delete(kx, 8, 1)
yx = np.array(dfx['class'])

dfe = pd.read_csv('Young_LBP.csv', header=None, names=names)
dje = pd.read_csv('Young_LBP.csv', header=None)
je = dje.values
ke = np.delete(je, 0, 1)
Xe = np.delete(ke, 8, 1)
ye = np.array(dfe['class'])


dj3 = pd.read_csv('Test_LBP.csv', header=None)
j3 = dj3.values
X3 = np.delete(j3, 0, 1)


X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.0, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.0, random_state=0)


X_test1 = np.concatenate((X1_test, X2_test), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)
X_train1 = np.concatenate((X1_train, X2_train), axis=0)
y_train = np.concatenate((y1_train, y2_train), axis=0)


X3_train, X3_test, y3_train, y4_test = train_test_split(Xx, ye, test_size=0.0, random_state=0)
X4_train, X4_test, y4_train, y3_test = train_test_split(Xx, ye, test_size=0.0, random_state=0)


X_test2 = np.concatenate((X3_test, X4_test), axis=0)
y_test2 = np.concatenate((y3_test, y4_test), axis=0)
X_train2 = np.concatenate((X3_train, X4_train), axis=0)
y_train2 = np.concatenate((y3_train, y4_train), axis=0)


XA=np.concatenate((X_train1, X_test1), axis=0)
YA=np.concatenate((y_train, y_test), axis=0)

XG=np.concatenate((X_train2, X_test2), axis=0)
YG=np.concatenate((y_train2, y_test2), axis=0)


knn = KNeighborsClassifier()
#knn = svm.SVC(kernel='linear', gamma=1)
knn.fit(XA, YA)
pred = knn.predict(X3)

knn1 = KNeighborsClassifier()
#knn1 = svm.SVC(kernel='linear', gamma=1)
knn1.fit(XG, YG)
pred1 = knn1.predict(X3)


clf1 = svm.SVC(kernel='linear', gamma=1)
clf2 = KNeighborsClassifier()
eclf = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2)], voting='hard')