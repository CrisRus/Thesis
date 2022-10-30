import cv2
import os

path = 'C:\\Users\\rus_c\\Desktop\\Thesis\\segmented\\train\\TURBERCULOSIS'
files = os.listdir(path)

for file in files:
    img = cv2.imread(path+'/'+file)
    resized = cv2.resize(img,(16,16))
    img_path = 'C:/Users/rus_c/Desktop/Thesis/resized/TURBERCULOSIS/'+file.split('.')[-2]+'.png'
    if cv2.imwrite(img_path, resized):
        print(img_path)
