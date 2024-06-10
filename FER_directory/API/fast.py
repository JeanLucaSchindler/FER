from fastapi import FastAPI, UploadFile, File
import numpy as np
import io
import cv2
from tensorflow.keras import models
from starlette.responses import Response
from FER_directory.logic_ml.preprocessing import process_image, decode_predictions, image_with_bounding_boxes

import os




app = FastAPI()

@app.post('/upload_image')
async def receive_image(img: UploadFile = File(...)):
    ### Receiving and decoding the image
    image = await img.read()

    nparr = np.fromstring(image, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # type(cv2_img) => numpy.ndarray


    X_image, boxes = process_image(cv2_img)

    # GET FER PREDICTIONS
    path_model = os.path.dirname(os.path.dirname(os.getcwd()))
    path_model = os.path.join(path_model,'models_trained/ResNet50V2_my_model_VM.h5')

    fer_model = models.load_model(path_model)
    predictions = fer_model.predict(X_image)

    #UPDATE ORIGINAL PHOTO WITH BOUNDING BOXES AND EMOTION LABEL

    labels = decode_predictions(predictions)



    buf = io.BytesIO()
    image_finale = image_with_bounding_boxes(cv2_img, boxes, labels)
    image_finale = np.array(image_finale)

    im = cv2.imencode('.jpg',image_finale)[1]

    return Response(content=im.tobytes(), media_type="image/png")
