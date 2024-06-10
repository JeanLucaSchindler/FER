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

    print(f'X_train.shape:{X_train.shape}')
    print(f'y_train.shape:{y_train.shape}')
    print(f'X_val.shape:{X_val.shape}')
    print(f'y_val.shape:{y_val.shape}')
    print(f'X_test.shape:{X_test.shape}')
    print(f'y_test.shape:{y_test.shape}')

    model = load_model()
    model.trainable = True
    model = add_last_layers(model)

    adam = Adam(learning_rate = 0.001)
    model = compile_model(model,adam)

    history = model_fit(model,X_train,y_train,X_val,y_val,batch_size =32, epochs = 40)

    print('Model training done✅')

    plot_history(history)

    print(model.evaluate(X_test,y_test)[-1])

    # y_pred = model.predict(X_test)
    # y_pred_final  = np.zeros_like(y_test)
    # y_pred_final[np.arange(len(y_pred_final)), np.argmax(y_pred, axis=-1)] = 1

    # target_names = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad','surprise']

    # print(classification_report(y_test, y_pred_final, target_names=target_names))

    model.save('ResNet50V2_my_model_VM.h5')
    model.save_weights('ResNet50V2_my_model_VM.weights.h5')
