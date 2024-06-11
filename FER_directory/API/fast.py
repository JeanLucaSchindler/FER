from fastapi import FastAPI, UploadFile, File
import numpy as np
import io
import cv2
from tensorflow.keras import models
from starlette.responses import Response
from starlette.responses import FileResponse
from FER_directory.logic_ml.preprocessing_photo import process_image, decode_predictions, image_with_bounding_boxes
from FER_directory.logic_ml.preprocessing_video import annotate_frame_with_boxes
import tempfile
import os

from keras.src.applications.inception_resnet_v2 import CustomScaleLayer
from keras.src.layers.layer import Layer
from tensorflow.keras import models

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
    path_model = os.path.join(path_model,'models_trained/model_chelou.h5')

    fer_model = models.load_model(path_model, custom_objects={'CustomScaleLayer': CustomScaleLayer})
    predictions = fer_model.predict(X_image/255)

    #UPDATE ORIGINAL PHOTO WITH BOUNDING BOXES AND EMOTION LABEL

    labels = decode_predictions(predictions)



    buf = io.BytesIO()
    image_finale = image_with_bounding_boxes(cv2_img, boxes, labels)
    image_finale = np.array(image_finale)

    im = cv2.imencode('.jpg',image_finale)[1]

    return Response(content=im.tobytes(), media_type="image/png")


@app.post('/upload_video')
async def receive_video(vid: UploadFile = File(...)):
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await vid.read())
        video_path = tmp.name

    # Create a temporary file for the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as temp_output:
        out_path = temp_output.name

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    # Load the pre-trained model
    path_model = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'models_trained', 'model_chelou.h5')
    fer_model = models.load_model(path_model, custom_objects={'CustomScaleLayer': CustomScaleLayer})

    frame_count = 0
    last_predictions = None
    last_boxes = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 'frame_skip' frames
        if frame_count % 16 == 0:
            faces, boxes = process_image(frame)
            if faces.size != 0:
                predictions = fer_model.predict(faces/255)
                labels = decode_predictions(predictions)
                last_predictions = labels
                last_boxes = boxes
            else:
                last_predictions = None
                last_boxes = None

        # Annotate the frame
        if last_predictions is not None and last_boxes is not None:
            annotated_frame = annotate_frame_with_boxes(frame, last_boxes, last_predictions)
        else:
            annotated_frame = frame

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Clean up the input video file
    os.remove(video_path)

    return FileResponse(out_path, media_type='video/avi', filename='processed_video.avi')
