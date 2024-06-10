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


def load_model():

    model = ResNet50V2(
      include_top=False ,
      weights='imagenet',
      input_tensor=None,
      input_shape=(96,96,3),
      pooling=None,
      classes=1000,
      classifier_activation='softmax'
    )

    print('Load model done✅')
    return model

def add_last_layers(model):
    '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''


    model = models.Sequential([
                model,
                Dropout(0.5),
                GlobalAveragePooling2D(),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(8, activation='softmax')
              ])

    print('Add_last_layers done✅')

    return model

def compile_model(model, optimizer_name = 'adam' ):
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer_name,
        metrics = ['accuracy']
      )
    print('Compile_model done✅')

    return model

def plot_history(history, title='', axs=None, exp_name=""):
    if axs is not None:
        ax1, ax2 = axs
    else:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    if len(exp_name) > 0 and exp_name[0] != '_':
        exp_name = '_' + exp_name
    ax1.plot(history.history['loss'], label='train' + exp_name)
    ax1.plot(history.history['val_loss'], label='val' + exp_name)
    #ax1.set_ylim(0., 2.2)
    ax1.set_title('loss')
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='train accuracy'  + exp_name)
    ax2.plot(history.history['val_accuracy'], label='val accuracy'  + exp_name)
    #ax2.set_ylim(0.25, 1.)
    ax2.set_title('Accuracy')
    ax2.legend()
    return (ax1, ax2)


def model_fit(model,X_train,y_train,X_val,y_val,batch_size,epochs):
    es = EarlyStopping(monitor = 'val_accuracy',patience = 7, restore_best_weights = True)

    Reducing_LR = ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.3,
                                                  patience=2,
                                                  verbose=1)

    datagen = ImageDataGenerator(
            rotation_range = 20,
            horizontal_flip = True
        )

    datagen.fit(X_train)
    train_generator = datagen.flow(X_train,y_train,batch_size, shuffle=True)

    print('Train_generator with augmented images done✅')


    # history = model.fit(
    #       train_generator,
    #       validation_data = (X_val,y_val),
    #       epochs = epochs,
    #       callbacks = [es, Reducing_LR],
    #       verbose = 1
    #     )

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val,y_val),
        shuffle = True,
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [es, Reducing_LR],
        verbose = 1
        )

    return history
