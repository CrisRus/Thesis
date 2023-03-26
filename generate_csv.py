import os
import pandas as pd
import numpy as np
from cv2 import imread
from cv2 import IMREAD_GRAYSCALE
from sklearn import preprocessing

target = []
flat_data = []
images = []
DataDirectory = 'segmented/train'

# Images to be classified as:
Categories = ["COVID19","NORMAL","PNEUMONIA", "TUBERCULOSIS"]

for i in Categories:
  print("Category is:",i,"\tLabel encoded as:",Categories.index(i))
  # Encode categories 
  target_class = Categories.index(i)
  # Create data path for all folders under Project
  path = os.path.join(DataDirectory,i)
  for img in os.listdir(path):
    if img.endswith('png'):
      img_array = imread(os.path.join(path,img), IMREAD_GRAYSCALE)
      print(img)
      flat_data.append(img_array.flatten())
      target.append(target_class)
# Convert list to numpy array format
flat_data = np.array(flat_data)
target = np.array(target)


df = pd.DataFrame(flat_data)
df[df.columns] = preprocessing.MinMaxScaler().fit_transform(df.values)
# Create a column for output data called Target
df['Target'] = target

df.to_csv("./segmented2.csv", index=False)

# Category is: COVID19            Label encoded as: 0
# Category is: NORMAL             Label encoded as: 1
# Category is: PNEUMONIA          Label encoded as: 2
# Category is: TURBERCULOSIS      Label encoded as: 3