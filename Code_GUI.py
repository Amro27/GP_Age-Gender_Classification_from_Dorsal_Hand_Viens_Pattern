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


# Local Ternary Pattern 3x3
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


def load_data(imageDir=[], image_path_list=[], image_list=[]):
    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        image_path_list.append(os.path.join(imageDir, file))
        Dir = os.path.join(imageDir, file)

    for imagePath in image_path_list:
        image = cv2.imread(str(imagePath), 0)
        image_list.append(image)


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
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 7)
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
        y = LocalTernaryPattern(images, 10)
        LTP.append(y)


def Apply_LBP(image_preprocessing=[], LBP=[]):
    for images in image_preprocessing:
        y = local_binary_pattern(images, 8, 1, method='uniform')
        LBP.append(y)


# Feature vectors from co occurrence Matrix
def create_csv_file_con(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['contrast']
        wr = csv.writer(resultFile, dialect='excel')
        for glcm in list1:
            hist1 = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
            f1 = []
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_dis(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['dissimilarity']
        wr = csv.writer(resultFile, dialect='excel')
        for glcm in list1:
            hist1 = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
            f1 = []
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_cor(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['correlation']
        wr = csv.writer(resultFile, dialect='excel')
        for glcm in list1:
            hist1 = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
            f1 = []
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_ASM(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        properties = ['ASM']
        wr = csv.writer(resultFile, dialect='excel')
        for glcm in list1:
            hist1 = np.hstack([greycoprops(glcm, prop).ravel() for prop in properties])
            f1 = []
            f11 = (np.sum(list(hist1), axis=0)) / (len(list1))
            f1.append(f11)
            # f1=list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)

            # feature vectors from histogram


def create_csv_file(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for image in list1:
            hist1, b1 = exposure.histogram(image, nbins=8)
            f1 = []
            f1 = list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


def create_csv_file_LBP(file_name, x, list1=[]):
    with open(file_name, 'wb') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        for im in list1:
            eps = 1e-7
            (hist, _) = np.histogram(im.ravel(),
                                     bins=np.arange(0, 8 + 3), range=(0, 8 + 2))

            # normalize the histogram
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            hist1 = hist
            f1 = []
            f1 = list(hist1)
            f1.append(x)
            k1 = np.asarray(f1)
            wr.writerow(k1)


# Read and Load Images
imageDirO = "F:/Database/Final_data/Age/Old"
image_path_listO = []
image_listO = []
load_data(imageDirO, image_path_listO, image_listO)

# Detect Region of Interest
image_listOROI = []
region_of_interest(image_listO, image_listOROI)

# Apply Bilateral filter
image_listOBL = []
bilateral_filter(image_listOROI, image_listOBL)

# Apply Morphology filter
image_listOmor = []
morphology_filter(image_listOBL, image_listOmor)

# Apply Median filter
image_listOmB = []
median_filter(image_listOmor, image_listOmB)

# Apply CLAHE
image_listOcl = []
CLAHE(image_listOmB, image_listOcl)

# Apply Adabtive threshold
image_preprocessingO = []
image_preprocessing(image_listOcl, image_preprocessingO)

# Apply Local Binary Pattern
LBPO = []
Apply_LBP(image_preprocessingO, LBPO)

# Apply Local Ternary Pattern
LTPO = []
Apply_LTP(image_preprocessingO, LTPO)

# Apply co occurrence matrix

TRY_comat = []
Apply_comatrix(image_preprocessingO, TRY_comat)

TRY_LBPO = []
Apply_comatrix(LBPO, TRY_LBPO)

TRY_LTPO = []
Apply_comatrix(LTPO, TRY_LTPO)

# Create female feature vectors
create_csv_file_con("Test_contrast.csv", None, TRY_comat)
create_csv_file_dis("Female_dissimilarity.csv", z, TRY_comat)
create_csv_file_cor("Female_correlation.csv", -1, TRY_comat)
create_csv_file_ASM("Female_ASM.csv", -1, TRY_comat)
create_csv_file_con("Female_contrast_LTP.csv", -1, TRY_LTPO)
create_csv_file_dis("Female_dissimilarity_LTP.csv", -1, TRY_LTPO)
create_csv_file_cor("Female_correlation_LTP.csv", -1, TRY_LTPO)
create_csv_file_ASM("Female_ASM_LTP.csv", -1, TRY_LTPO)
create_csv_file_con("Female_contrast_LBP.csv", -1, TRY_LBPO)
create_csv_file_dis("Female_dissimilarity_LBP.csv", -1, TRY_LBPO)
create_csv_file_cor("Female_correlation_LBP.csv", -1, TRY_LBPO)
create_csv_file_ASM("Female_ASM_LBP.csv", -1, TRY_LBPO)
create_csv_file("Female_LTP.csv", -1, LTPO)
create_csv_file("Female_LBP.csv", -1, LBPO)
create_csv_file_LBP("Female_normalized_LBP.csv", -1, LBPO)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# Load feature vectors
names = ['class']
df1 = pd.read_csv('Young_dissimilarity.csv', header=None, names=names)
dj1 = pd.read_csv('Young_dissimilarity.csv', header=None)
j1 = dj1.values
X1 = np.delete(j1, 1, 1)
y1 = np.array(df1['class'])

df2 = pd.read_csv('Old_dissimilarity.csv', header=None, names=names)
dj2 = pd.read_csv('Old_dissimilarity.csv', header=None)
j2 = dj2.values
X2 = np.delete(j2, 1, 1)
y2 = np.array(df2['class'])

# Normalization and Preperation of feature vectors (Not Used)
# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaledX1 = scaler.fit_transform(X1)
# rescaledX2 = scaler.fit_transform(X2)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.5, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=0)

X_test = np.concatenate((X1_train, X2_train), axis=0)
y_test = np.concatenate((y1_train, y2_train), axis=0)
X_train = np.concatenate((X1_test, X2_test), axis=0)
y_train = np.concatenate((y1_test, y2_test), axis=0)

# creating odd list of K for KNN
myList = list(range(1, 12))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

good = 0
bad = 0
acc = 0
ac = []

# Apply many classifiers
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    # knn = svm.SVC(kernel='linear', C=k, gamma=1)
    # knn = GaussianNB()
    # knn = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    # knn = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    for i in range(0, len(pred)):
        if pred[i] == y_test[i]:
            good += 1
        else:
            bad += 1
    acc = ((float(good) / len(pred)) * 100)
    ac.append(acc)
    # print acc
    good = 0
    bad = 0
print sum(ac) / 6