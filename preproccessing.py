import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
# Split data into input and output sets
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from PIL import Image
from sklearn.ensemble import RandomForestClassifier


# x is all input values of images and their pixel values (90 images * 67500)
# y is output values or correct label of image (90 images * 1 column of labels)


target = []
flat_data = []
images = []
DataDirectory = '/Users/darren/python_recognization/Classification_Images'

# Images to be classified as:
Categories = ["crowd_cheering","smile","wave"]

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Flatten the axes (axs) if you want to iterate through it in a single loop
axs = axs.flatten()
img_idx = 0

for i in Categories:
  print("Category is:",i,"\tLabel encoded as:",Categories.index(i))
  # Encode categories cute puppy as 0, icecream cone as 1 and red rose as 2
  target_class = Categories.index(i)
  # Create data path for all folders under MinorProject
  path = os.path.join(DataDirectory,i)
  # Image resizing, to ensure all images are of same dimensions
  img_shown = False

  for img in os.listdir(path):
    # img_array = imread(os.path.join(path,img))
    img_array = Image.open(os.path.join(path, img))

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
df['Target'] = target
# Rows are all the input images (90 images, 30 of each category)
df
print(plt.imshow(images[20]))
x = df.iloc[:,:-1].values
y = target
print("Input data dimensions:",x.shape)
print("Output data dimensions:",y.shape)

# Stratify ensures every image is divided in equal proportions (no bias)
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True,test_size = 0.3,random_state=109,stratify=y)
print("Dimensions of input training data:",x_train.shape)
print("Dimensions of input testing data:",x_test.shape)
print("Dimensions of output training data:",y_train.shape)
print("Dimensions of output testing data:",y_test.shape)
print("Labels\t\t   Image index considered")
print(np.unique(y_train,return_counts=True))
print(np.unique(y_test,return_counts=True))

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
                    
# Apply GridSearchCV to find best parameters for given dataset
# verbose is used to describe the steps taken to find best parameters
#cv = GridSearchCV(SVC(), tuned_parameters, refit = True,verbose= 3)
cv = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0) 
cv.fit(x_train,y_train)
# Save the model to a file.
filename = "model.joblib"
dump(cv, open(filename, "wb"))

# Load the model from a file.
#model = load(open(filename, "rb"))