from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os
import numpy as np
from PIL import Image
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
img = "capture.ppm"
path = "/home/pi/python_recognization/face_image.jpg"
# img_array = imread(os.path.join(path,img))
img_array = Image.open(path)
img_shown=False
flat_data = []
images = []
target = []
target_class = []
img_idx=0
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()

if img_array.mode != 'RGB':
    img_array = img_array.convert('RGB')
img_array = np.array(img_array)
if not img_shown:
  print('showing image ', img_idx)
  # plt.imshow(img_array)
  axs[img_idx].imshow(img_array)
  img_idx += 1
  img_shown = True
# Skimage normalizes the value of image
img_resized = resize(img_array,(150,150,3))[0]
flat_image = img_resized.flatten()
print(flat_image.shape)
flat_data.append(flat_image)
images.append(img_resized)
target.append(target_class)
# Convert list to numpy array format
flat_data = np.array(flat_data)
images = np.array(images)
target = np.array(target)
df = pd.DataFrame(flat_data)
# Create a column for output data called Target
#df['Target'] = target
# Rows are all the input images (90 images, 30 of each category)
print(df)
model = load("model.joblib")
print(
model.predict(df))
