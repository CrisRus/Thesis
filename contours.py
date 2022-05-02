import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from contrast import increase_contrast

path = 'train/COVID19'
files = os.listdir(path)


for file in files:
    # img = cv2.imread(path+'/'+file)
    # incresead contrast because we want to use edge detection, not much diff between the background and the lungs themselves so we needed to increase the contrast
    img = increase_contrast(path+'/'+file)
    img = cv2.resize(img,(256,256))

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    threshold = np.average(gray)
    _,thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_TOZERO_INV)
    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)
    # generate the mask, mask is all black except where the lungs is white. 
    mask = np.zeros((256,256), np.uint8)
    img1 = cnt[-1]
    masked1 = cv2.drawContours(mask, [img1],-1, 255, -1)
    try:
        img2 = cnt[-2]
        masked2 = cv2.drawContours(mask, [img2],-1, 255, -1)
        masked = cv2.bitwise_and(masked1, masked2)
    except Exception as e:
        masked = masked1
        print(e)

    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    # plt.imshow(segmented)
    # plt.show()
    img_path = 'D:\lungs\segmented/'+path+'/'+'contrasted'+file.split('.')[-2]+'.png'
    if cv2.imwrite(img_path, segmented):
        print(img_path)

