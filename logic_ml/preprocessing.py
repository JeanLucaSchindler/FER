import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import os
from sklearn.preprocessing import LabelEncoder


def get_image(path):
    '''
    receives a path of image,
    and returns image in array form
    '''
    image = Image.open(path)

    return np.array(image) / 255

def preproc_train():
    '''
    receives a csv,
    and returns array of all images (in array form) in that csv
    '''
    path_csv =os.path.dirname(os.getcwd())
    path_csv = os.path.join(path_csv+'/raw_data/labels.csv')
    data = pd.read_csv(path_csv)

    # Initialize an empty list to hold the images
    images = []

    # Iterate over the paths and load each image
    for path in data['pth']:
        raw_data_path = os.path.dirname(os.getcwd())
        raw_data_path = os.path.join(raw_data_path+'/raw_data/'+path)
        images.append(get_image(raw_data_path))

    # Convert the list of images to a numpy array
    X = np.stack(images)

    y = np.array(data.label)

    return X, y

def label_categorize(y):
    '''
    receives y and
    categorizes using to_categorical
    '''
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder and transform the labels to numerical labels
    y_encoded = label_encoder.fit_transform(y)

    # Convert numerical labels to one-hot encoded format
    y_cat = to_categorical(y_encoded)
    return y_cat

def preproc(path):
    image = Image.open(path)
    image = image.resize((96,96))
    return np.array(image) / 255
