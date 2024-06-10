import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras import layers


if __name__ == "__main__":

    path = os.path.join(os.path.dirname(os.getcwd()),'data_preprocessed')

    # X = np.load(os.path.join(path,'X.npy'))
    # y_cat = np.load(os.path.join(path,'y_cat.npy'))
    # print(f'X.shape:{X.shape}')
    # print(f'y_cat.shape:{y_cat.shape}')

    # X_train,X_test,y_train,y_test = train_test_split(X,y_cat,stratify=y_cat,test_size = 0.2)
    # X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,stratify=y_train,test_size = 0.2)

    model = models.load_model(
        "ResNet50V2_my_model_kaggle.h5")
        # custom_objects={"CustomScaleLayer": CustomScaleLayer}

    X_test=np.load('X_test.npy')
    y_test= np.load('y_test.npy')

    print(model.evaluate(X_test,y_test)[-1])
    print(model.predict(X_test))
