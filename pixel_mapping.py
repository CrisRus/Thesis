from operator import index
import numpy as np
from kneed import KneeLocator
import os
import cv2


path = 'segmented/test/{}'


def mark_image(indexes, label):
    files = os.listdir(path.format(label))
    print(len(indexes))
    for file in files:
        img = cv2.imread(path.format(label)+'/'+file, 1)
        for index in indexes:
            img[index//256, index%256] = (0, 0, 255)
        cv2.imwrite("marked/{}/{}".format(label, file), img)



def map_pixels(coef, labels):
    x_axis = [i for i in range(65536)]
    elbow = []
    for i in range(0, 4):
        kn = KneeLocator(x_axis, sorted(coef[i], reverse=True), curve='convex', direction='decreasing')
        elbow.append(kn.knee)
    indexes = [[], [], [], []]
    for i in range(0, 4):
        important_coefs = sorted(coef[i], reverse=True)[0: np.max(elbow)]
        for c in important_coefs:
            indexes[i].append(list(coef[i]).index(c))
        mark_image(indexes[i], labels[i])