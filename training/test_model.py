import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_build import load_model, add_last_layers
from model_build import compile_model,plot_history,model_fit



if __name__ == "__main__":

    path = os.path.join(os.path.dirname(os.getcwd()),'data_preprocessed')

    X = np.load(os.path.join(path,'X.npy'))
    y_cat = np.load(os.path.join(path,'y_cat.npy'))
    print(f'X.shape:{X.shape}')
    print(f'y_cat.shape:{y_cat.shape}')

    X_train,X_test,y_train,y_test = train_test_split(X,y_cat,stratify=y_cat,test_size = 0.2)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,stratify=y_train,test_size = 0.2)


    model = models.load_model("ResNet50V2_my_model.h5")

    print(model.predict(X_test))
