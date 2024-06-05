import streamlit as st
from altair.vegalite.v4.api import Chart

import numpy as np
import pandas as pd

import random

import time

from PIL import Image
from logic_ml.preprocessing import get_image
import matplotlib.pyplot as plt

st.markdown("""# Designing a Facial Emotion Recognition model
## Want a new set of pictures?
### Move the cursor and observe""")

df = pd.DataFrame({
    'first column': list(range(1, 11)),
    'second column': np.arange(10, 101, 10)
})

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('', 1, 5, 3)

# and used to select the displayed lines
#head_df = df.head(line_count)

#head_df


data = pd.read_csv('/Users/marinelegall/code/marinoulg/FER_Project/FER/raw_data/labels.csv')
my_random = random.randrange(0,len(data['pth']))


labels_df = list(data['label'].unique())




def emotions():
    my_images = []
    captions = []
    indices = []

    for label in labels_df:
        emotion = pd.DataFrame(data[data['label']==label]).reset_index()
        emotion_index = emotion.index
        my_random_emotion = random.randrange(emotion_index[0], emotion_index[-1])
        image = Image.open('raw_data/'+emotion['pth'][my_random_emotion])
        my_images.append(image)
        label = (emotion['label'][my_random_emotion])
        captions.append(label)
        indices.append(my_random_emotion)

    return st.image(my_images, caption=captions, width=200)

#emotions()

def function_emotion(emotion):
    """
    Create a function that returns a random image and its label for a given emotion in
    our dataset (aka happy, sad, fear..., which are labels)
    """
    df_emotion = pd.DataFrame(data[data['label']==emotion]).reset_index()
    df_emotion_index = df_emotion.index
    my_random_df_emotion = random.randrange(df_emotion_index[0], df_emotion_index[-1])
    image = Image.open('raw_data/'+df_emotion['pth'][my_random_df_emotion])
    label = (df_emotion['label'][my_random_df_emotion])
    return image,label

image_sad, label_sad = function_emotion('sad')

def get_my_images_and_their_label(labels):
    """
    Create a function that couples all emotions into 1 big block and
    returns a list of images (for different emotions) along with their
    associated label
    Attention: it does the same thing as emotions()
    """
    my_images = []
    my_labels = []

    for label in labels:
        image, label = function_emotion(label)
        my_images.append(image)
        my_labels.append(label)

    return my_images, my_labels



my_images, my_labels = get_my_images_and_their_label(labels_df)
#my_images_2, my_labels_2 = get_my_images_and_their_label()

st.image(my_images, my_labels, width=200)
#st.image(my_images_2, my_labels_2, width=200)



def tourner_en_boucle(image, label):
    for _ in range(2):
        with st.empty():
            for seconds in range(3):
                #st.write(f":hourglass_flowing_sand: {seconds} seconds have passed")
                #time.sleep(1)
                a = st.image(image, label, width=200)
    return a

#tourner_en_boucle(image_neutral, label_neutral)


_LOREM_IPSUM = """
Lorem ipsum dolor sit amet, **consectetur adipiscing** elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
"""


def stream_data(lorem_ipsum):
    for word in lorem_ipsum.split(" "):
        yield word + " "
        time.sleep(0.5)

    # yield pd.DataFrame(
    #     np.random.randn(5, 10),
    #     columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    # )

    for word in lorem_ipsum.split(" "):
        yield word + " "
        time.sleep(0.02)
    return st.write(stream_data)


st.text((_LOREM_IPSUM ))

def stream_image(my_images):
    for image in my_images:
        yield image
        time.sleep(0.5)

    # yield pd.DataFrame(
    #     np.random.randn(5, 10),
    #     columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
    # )

    return image


stream_image(my_images)
