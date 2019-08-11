import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize    
from skimage.color import rgb2gray
import random
import cv2

imageSize = 300
count = 0

for item in os.listdir('testset/images/'):
    imageName = item[:-4]
    print(imageName)
    array = None

    #Image
    try:
        image = matplotlib.image.imread('testset/images/' + imageName + '.png')
    except Exception:
        image = matplotlib.image.imread('testset/images/' + imageName + '.jpg')
    #Points
    with open('testset/imageValidation/' + imageName + '.pts') as p:
        rows = [rows.strip() for rows in p]

    head = rows.index('{') + 1
    tail = rows.index('}')

    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]
    points = [tuple([float(point) for point in coords]) for coords in coords_set]

    for x in points:
        try:
            array = np.append(array,[int(x[0]),int(x[1])], axis=0)
        except Exception:
            array = np.array([int(x[0]),int(x[1])])
    
    xMax = max(array[0::2]) + random.randint(10,30)
    yMax = max(array[1::2]) + random.randint(10,30)
    xMin = min(array[0::2]) - random.randint(10,20)
    yMin = min(array[1::2]) - random.randint(10,20)
    
    if xMin < 0 or xMax == 0:
        xMin = 1
    if yMin < 0 or yMin == 0:
        yMin = 1
    
    array[0::2] = array[0::2]-xMin
    array[1::2] = array[1::2]-yMin
    
    print(array.shape)
    
    try:
        if image.shape[2] != 3:
            continue
    except Exception:
        continue
    
    cImage = image[ yMin:yMax, xMin:xMax, 1]
    
    nX = imageSize/cImage.shape[1]
    nY = imageSize/cImage.shape[0]
    
    array[0::2] = array[0::2] * nX
    array[1::2] = array[1::2] * nY
    
    cImage = resize(cImage, (imageSize, imageSize))
    cImage = rgb2gray(cImage)
    cImage = cImage.reshape(cImage.shape[0], cImage.shape[1], 1)

    if True:
        try:
            imageCollection = np.append(imageCollection, np.array([cImage]), axis = 0)
            pointCollection = np.append(pointCollection, np.array([array]), axis = 0)
        except Exception:
            imageCollection = np.array([cImage])
            pointCollection = np.array([array])
        
        if count % 400 == 0:
            np.save('testX.npy', imageCollection)
            np.save('testY.npy', pointCollection)
    else:
        imageCollection = np.array([cImage])
        pointCollection = np.array([array])
    
    count += 1

np.save('testX.npy', imageCollection)
np.save('testY.npy', pointCollection)    