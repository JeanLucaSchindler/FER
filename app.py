import streamlit as st
from altair.vegalite.v4.api import Chart
import numpy as np
import pandas as pd
import random
import os
from PIL import Image
import time


st.markdown("""# This is a header
## This is a sub header
This is text""")

df = pd.DataFrame({
    'first column': list(range(1, 11)),
    'second column': np.arange(10, 101, 10)
})

# this slider allows the user to select a number of lines
# to display in the dataframe
# the selected value is returned by st.slider
line_count = st.slider('Select a line count', 1, 10, 3)

# and used to select the displayed lines
head_df = df.head(line_count)

head_df

path_csv =os.path.dirname(os.getcwd())
path_csv = os.path.join(path_csv+'/raw_data/labels.csv')
data = pd.read_csv(path_csv)

with st.empty():
    for seconds in range(60):
        st.write(f"⏳ {seconds} seconds have passed")
        time.sleep(1)
    st.write("✔️ 1 minute over!")



image = Image.open('raw_data/'+data['pth'][2])
image_2 = Image.open('raw_data/'+data['pth'][3])
st.image([image, image_2], width=200, caption=['Anger', 'disgust'])
