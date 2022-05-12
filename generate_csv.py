import os
import pandas as pd
import numpy as np
from cv2 import imread
from skimage.transform import resize
from sklearn import preprocessing

target = []
flat_data = []
images = []
DataDirectory = 'C:\\Users\\Cristina\\Desktop\\Thesis\\segmented\\train'

# Images to be classified as:
Categories = ["COVID19","NORMAL","PNEUMONIA", "TURBERCULOSIS"]

for i in Categories:
  print("Category is:",i,"\tLabel encoded as:",Categories.index(i))
  # Encode categories cute puppy as 0, icecream cone as 1 and red rose as 2
  target_class = Categories.index(i)
  # Create data path for all folders under Project
  path = os.path.join(DataDirectory,i)
  # Image resizing, to ensure all images are of same dimensions
  for img in os.listdir(path):
    img_array = imread(os.path.join(path,img))
    # Skimage normalizes the value of image
    img_resized = resize(img_array,(100,100,3))
    flat_data.append(img_resized.flatten())
    target.append(target_class)
# Convert list to numpy array format
flat_data = np.array(flat_data)
target = np.array(target)


df = pd.DataFrame(flat_data)
df[df.columns] = preprocessing.MinMaxScaler().fit_transform(df.values)
# Create a column for output data called Target
df['Target'] = target

df.to_csv("./train_dataset.csv")