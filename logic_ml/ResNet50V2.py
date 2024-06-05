#Imports
from PIL import Image

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.applications import ResNet50V2

from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Nadam


# get preproc data
""" - if data preproc exists (aka if X.py and y_cat.py exist) --> do not preproc but NEED TO NORMALIZE
    - else: preproc
"""


#load my X and y_cat preprocessed
X = np.load('X.npy')
y_cat = np.load('y_cat.npy')

X.shape, y_cat.shape

#convert X to float32
X = X.astype('float32')

# Normalizing X
X = X/255

#train/test split
X_train,X_test,y_train,y_test = train_test_split(X,y_cat,stratify=y_cat,test_size = 0.2)

#train/val split
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,stratify=y_train,test_size = 0.2)


#Loading pretrained model

# include_top=False  --> removes the top part of the neural network
# classes --> can be removed as include_top=False

def load_model():

    model = ResNet50V2( # ResNet50V2 --> check le reste
      include_top=False ,
      weights='imagenet',
      input_tensor=None,
      input_shape=X_train.shape[1:],
      pooling=None,
      classes=1000,
      classifier_activation='softmax'
    )
    return model


# Add last layers of pretrained model ResNet50V2
def add_last_layers(model):
    '''Takes a pre-trained model, sets its parameters to non-trainable, and adds additional trainable layers on top'''

    model = models.Sequential([
            model,
            GlobalAveragePooling2D(),
            Dense(254, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='softmax')
                ])


    return model

# compile model
def compile_model(model, optimizer_name = 'adam' ):
    '''return a compiled model suited for the CIFAR-10 task'''
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = optimizer_name,
        metrics = ['accuracy']
      )

    return model

# Function to plot val_loss and val_accuracy
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

# Loading model
model = load_model()

model.summary()

# Freezing the parameters of the CNN part
model.trainable = True

# Adding last layers
model = add_last_layers(model)

model.summary()

#we use to optimizer Nadam
nadam = Nadam(learning_rate = 0.001)
model = compile_model(model,nadam)

#Let's deal with the hyperparams of callbacks (within the model.fit())

# Create Early Stopping Callback to monitor the accuracy
Early_Stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7, restore_best_weights = True, verbose=1)

# Create ReduceLROnPlateau Callback to reduce overfitting by decreasing learning
Reducing_LR = ReduceLROnPlateau(monitor='val_loss',
                                factor=0.2,
                                patience=2,
                                verbose=1)

callbacks = [Early_Stopping, Reducing_LR]

history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val,y_val),
        shuffle = True,
        batch_size = 16,
        epochs = 100,
        callbacks = callbacks,
        verbose = 1
    )

#Plot the history to compare val_accuracy and val_loss
plot_history(history)

#Evaluate model
model.evaluate(X_test,y_test)[-1]
# accuracy = model.evaluate(X_test,y_test)[-1]
# accuracy

y_pred = model.predict(X_test)
