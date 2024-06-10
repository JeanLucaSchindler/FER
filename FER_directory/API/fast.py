#import requests

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

import numpy as np
#import cv2
import io
#from face_rec.face_detection import annotate_face

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict')
def predict():
    return {'wait': 64}


# url = 'http://localhost:8000/predict'

# params = {
#     'day_of_week': 0, # 0 for Sunday, 1 for Monday, ...
#     'time': '14:00'
# }

# response = requests.get(url, params=params)
# res = response.json() #=> {'wait': 64}
# prediction = res['wait']

# @app.post('/upload_image')
# async def receive_image(img: UploadFile=File(...)):
#     ### Receiving and decoding the image
#     contents = await img.read()

#     nparr = np.fromstring(contents, np.uint8)
#     cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray

#     ### Do cool stuff with your image.... For example face detection
#     #annotated_img = annotate_face(cv2_img)

#     ### Encoding and responding with the image
#     im = cv2.imencode('.png')[1] # extension depends on which format is sent from Streamlit
#     return Response(content=im.tobytes(), media_type="image/png")

@app.post('/upload_image')
def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = img.read()
    return contents
