import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras import models
import os
import cv2
import mediapipe as mp

# PREPROCESS VIDEO DATA

def define_faces(image):
    mp_face_detection = mp.solutions.face_detection
    ih, iw, _ = image.shape
    bounding_boxes = []

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * iw)
            y_min = int(bboxC.ymin * ih)
            width = int(bboxC.width * iw)
            height = int(bboxC.height * ih)
            x_max = x_min + width
            y_max = y_min + height
            bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes

def crop_image(image, boxes):
    list_faces = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cropped_image = image[y_min:y_max, x_min:x_max]
        list_faces.append(cv2.resize(cropped_image, (96, 96)))
    if list_faces:
        return np.stack(list_faces)
    else:
        return np.array([])

def process_image(image):
    boxes = define_faces(image)
    faces = crop_image(image, boxes)
    return faces, boxes

def annotate_frame_with_boxes(frame, boxes, texts):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, texts[i], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    return frame

def decode_predictions(predictions):

    y_pred = np.argmax(predictions, axis=-1)

    labels_dict = {0: 'anger',
                   1: 'contempt',
                   2: 'disgust',
                   3: 'fear',
                   4: 'happy',
                   5: 'neutral',
                   6: 'sad',
                   7: 'surprise'}

    labels = []
    for pred in y_pred:
        labels.append(labels_dict[pred])

    return labels


# CHOOSE AMOUNT OF PREDICTIONS PER SECOND (FULL OR SKIP FRAME)

def annotate_faces_in_video_light(input_video_path, output_video_path, model_path, frame_skip=16):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    fer_model = models.load_model(model_path)
    frame_count = 0
    last_predictions = None
    last_boxes = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process every 'frame_skip' frames
        if frame_count % frame_skip == 0:
            faces, boxes = process_image(frame)
            if faces.size != 0:
                predictions = fer_model.predict(faces)
                labels = decode_predictions(predictions)
                last_predictions = labels
                last_boxes = boxes
            else:
                last_predictions = None
                last_boxes = None
        else:
            if last_predictions is not None and last_boxes is not None:
                annotated_frame = annotate_frame_with_boxes(frame, last_boxes, last_predictions)
            else:
                annotated_frame = frame

        if last_predictions is not None and last_boxes is not None:
            annotated_frame = annotate_frame_with_boxes(frame, last_boxes, last_predictions)
        else:
            annotated_frame = frame

        out.write(annotated_frame)
        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def annotate_faces_in_video_full(input_video_path, output_video_path, model_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    fer_model = models.load_model(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        faces, boxes = process_image(frame)
        if faces.size == 0:
            out.write(frame)
            continue

        predictions = fer_model.predict(faces)
        labels = decode_predictions(predictions)
        annotated_frame = annotate_frame_with_boxes(frame, boxes, labels)
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
