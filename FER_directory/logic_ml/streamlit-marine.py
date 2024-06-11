import streamlit as st
from altair.vegalite.v4.api import Chart

import numpy as np
import pandas as pd
import random
import time
from PIL import Image
import matplotlib.pyplot as plt
import base64

# Set the background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Path to your background image
background_image_path = '/Users/marinelegall/Downloads/inside-out.jpeg'

# Set the background
set_background(background_image_path)

# Create a title with custom color
st.markdown(
    """
    <style>
    .title {
        color: white;
        font-size: 2em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the title
st.markdown('<h1 class="title">Designing a Facial Emotion Recognition model</h1>', unsafe_allow_html=True)

#st.markdown("""# Designing a Facial Emotion Recognition model
#""")

# df = pd.DataFrame({
#     'first column': list(range(1, 11)),
#     'second column': np.arange(10, 101, 10)
# })

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
#line_count = st.slider('', 1, 5, 3)

# # and used to select the displayed lines
# #head_df = df.head(line_count)

# #head_df


data = pd.read_csv('/Users/marinelegall/code/marinoulg/FER_Project/FER/raw_data/labels.csv')
my_random = random.randrange(0,len(data['pth']))


labels_df = list(data['label'].unique())

"""
Have pictures from batch #1672 appear on screen
"""
#st.image('/Users/marinelegall/code/marinoulg/FER_Project/FER/#1672-data-set/anger/IMG_20240606_114127.jpg')
df = pd.read_csv('/Users/marinelegall/code/marinoulg/FER_Project/FER/#1672-data-set/raw-data-wagon-CLEAN.csv', sep=';')

def emotions_1672():

    # Create a placeholder for images
    placeholder = st.empty()

    # Initialize current images and captions
    current_images = []
    current_captions = []

    # Initialize images for the first time
    for label in labels_df:
        emotion = df[df['label'] == label].reset_index()
        my_random_emotion = random.choice(emotion.index)
        image_path = '../../../#1672-data-set/' + emotion.loc[my_random_emotion, 'pth']
        image = Image.open(image_path)
        current_images.append(image)
        current_captions.append(label)

    while True:
        new_images = []
        new_captions = []

        for i, label in enumerate(labels_df):
            # Decide whether to change the image
            if random.random() < 0.5 and current_images[i] is not None:  # 50% chance to change the image
                emotion = df[df['label'] == label].reset_index()
                my_random_emotion = random.choice(emotion.index)
                image_path = '../../../#1672-data-set/' + emotion.loc[my_random_emotion, 'pth']
                image = Image.open(image_path)
                new_images.append(image)
                new_captions.append(label)
            else:
                new_images.append(current_images[i])
                new_captions.append(current_captions[i])

        # Update the current images and captions
        current_images = new_images
        current_captions = new_captions

        # Update the images in the placeholder
        placeholder.image(current_images, caption=current_captions, width=150)

        # Wait for 2 seconds before updating again
        #time.sleep(2)

# Call the function to start displaying images
emotions_1672()









# def emotions():
#     my_images = []
#     captions = []
#     indices = []

#     for label in labels_df:
#         emotion = pd.DataFrame(data[data['label']==label]).reset_index()
#         emotion_index = emotion.index
#         my_random_emotion = random.randrange(emotion_index[0], emotion_index[-1])
#         image = Image.open('../../raw_data/'+emotion['pth'][my_random_emotion])
#         my_images.append(image)
#         label = (emotion['label'][my_random_emotion])
#         captions.append(label)
#         indices.append(my_random_emotion)

#     return st.image(my_images, caption=captions, width=200)

#emotions()

def function_emotion(emotion):
    """
    Create a function that returns a random image and its label for a given emotion in
    our dataset (aka happy, sad, fear..., which are labels)
    """
    df_emotion = pd.DataFrame(df[df['label']==emotion]).reset_index()
    df_emotion_index = df_emotion.index
    my_random_df_emotion = random.randrange(df_emotion_index[0], df_emotion_index[-1])
    image = Image.open('../../#1672-data-set/'+df_emotion['pth'][my_random_df_emotion])
    label = (df_emotion['label'][my_random_df_emotion])
    return image,label



# image_sad, label_sad = function_emotion('sad')
# image_anger, label_anger = function_emotion('anger')
# image_happy, label_happy = function_emotion('happy')
# image_surprise, label_surprise = function_emotion('surprise')

# figu, axs = plt.subplots(2,2)
# axs[0,0].imshow(image_sad)
# time.sleep(1)
# axs[1,0].imshow(image_happy)
# time.sleep(1)
# axs[0,1].imshow(image_surprise)
# time.sleep(1)
# axs[1,1].imshow(image_anger)

# st.pyplot(figu)



# image_angry, label_angry = function_emotion('anger')

# plt.subplot(2,4,2)
# plt.imshow(image_angry)
# time.sleep(1)

#st.image([image_angry], [label_angry], width=200)

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



#my_images, my_labels = get_my_images_and_their_label(labels_df)
#my_images_2, my_labels_2 = get_my_images_and_their_label()

#st.image(my_images, my_labels, width=200)
# #st.image(my_images_2, my_labels_2, width=200)



# def tourner_en_boucle(image, label):
#     for _ in range(2):
#         with st.empty():
#             for seconds in range(3):
#                 #st.write(f":hourglass_flowing_sand: {seconds} seconds have passed")
#                 #time.sleep(1)
#                 a = st.image(image, label, width=200)
#     return a

# #tourner_en_boucle(image_neutral, label_neutral)


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

stream_data(_LOREM_IPSUM)

# st.text((_LOREM_IPSUM ))

# def stream_image(my_images):
#     for image in my_images:
#         yield image
#         time.sleep(0.5)

#     # yield pd.DataFrame(
#     #     np.random.randn(5, 10),
#     #     columns=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
#     # )

#     return image


# stream_image(my_images)



import streamlit as st
from PIL import Image
import requests
from dotenv import load_dotenv
import os


# # App title and description
# st.header('Simple Image Uploader 📸')
# st.markdown('''
#             > This is a Le Wagon boilerplate for any data science projects that involve exchanging images between a Python API and a simple web frontend.

#             > **What's here:**

#             > * [Streamlit](https://docs.streamlit.io/) on the frontend
#             > * [FastAPI](https://fastapi.tiangolo.com/) on the backend
#             > * [PIL/pillow](https://pillow.readthedocs.io/en/stable/) and [opencv-python](https://github.com/opencv/opencv-python) for working with images
#             > * Backend and frontend can be deployed with Docker
#             ''')

# st.markdown("---")

# ### Create a native Streamlit file upload input
# st.markdown("### Let's do a simple face recognition 👇")
# img_file_buffer = st.file_uploader('Upload an image')

if img_file_buffer is not None:

  col1, col2 = st.columns(2)

  with col1:
    ### Display the image user uploaded
    st.image(Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️")

  with col2:
    with st.spinner("Wait for it..."):
      ### Get bytes from the file buffer
      img_bytes = img_file_buffer.getvalue()

      ### Make request to  API (stream=True to stream response as bytes)
      res = requests.post(url + "/upload_image", files={'img': img_bytes})

      if res.status_code == 200:
        ### Display the image returned by the API
        st.image(res.content, caption="Image returned from API ☝️")
      else:
        st.markdown("**Oops**, something went wrong 😓 Please try again.")
        print(res.status_code, res.content)




# -----------------------------------TESTS------------------------------------------------
# #have a video (from my computer) displayed on Streamlit
# video_file = open('/Users/marinelegall/Downloads/VID_20240531_224913.mp4', 'rb')
# video_bytes = video_file.read()
# st.video(video_bytes)

# #allow to drag and drop photo from computer
# #st.file_uploader(label='Drag/Drop your photo here')

# #allow webcam --> to take a photo
# st.camera_input(label='This is me')
