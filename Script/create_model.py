import os
import cv2
import pandas as pd
# import pickle
import numpy as np
import tensorflow as tf
# import seaborn as sns
# from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
import matplotlib.pyplot as plt
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.layers import Dropout, Flatten, Dense
# from keras.models import Sequential
# from keras.utils.vis_utils import plot_model
# from keras.callbacks import ModelCheckpoint
# from keras.utils import to_categorical
# from sklearn.metrics import confusion_matrix
# from keras.preprocessing import image
# import tensorflow as tf
from PIL import Image
from imutils import paths


def img_df(image_path, shape):
    image = Image.open(image_path)
    image_resized = image.resize(shape, Image.ANTIALIAS)
    img_array = np.asarray(image_resized)
    return img_array


path_train_data = './Raw-Data/state-farm-distracted-driver-detection/imgs/train'
classes = os.listdir(path_train_data)
numero_classes = len(classes)
image_dimensions = (32, 32, 3)

images = []
class_ = []
for class_name in classes:
    img_list = sorted(list(paths.list_images(f'{path_train_data}\{class_name}')))
    for img_name in img_list:
        try:
            img = cv2.imread(img_name)
            img = cv2.resize(img, (224, 224))
            images.append(img)
            class_.append(class_name)
        except Exception as e:
            print(f'error {e}')

images = np.array(images)
class_ = np.array(class_)
print(len(images), len(class_))
df = pd.DataFrame()
df['images'] = list(images)
df['Y'] = list(class_)
print(df.head(5))

X_train, X_validation, y_train, y_validation = train_test_split(images, class_, test_size=0.2)

print('check_number_images_class')
num_examples = []
for class_name in classes:
    num = len(np.where(y_train == class_name)[0])
    num_examples.append(num)

plt.figure(figsize=(8, 5))
plt.bar(x=classes, height=num_examples)
plt.title('Num of images for each Class')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.show()

SEED = 1534

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.1, seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomZoom((0.0, 0.3), seed=SEED),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, seed=SEED),
    ]
)
