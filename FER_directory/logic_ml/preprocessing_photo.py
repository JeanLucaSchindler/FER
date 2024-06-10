import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import os
from sklearn.preprocessing import LabelEncoder
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# PREPROCESSING FOR TRAIN DATA

def get_image(path):
    '''
    receives a path of image,
    and returns image in array form
    '''
    image = Image.open(path)

    return np.array(image) / 255

def preproc_train():
    '''
    receives a csv,
    and returns array of all images (in array form) in that csv
    '''
    path_csv =os.path.dirname(os.getcwd())
    path_csv = os.path.join(path_csv+'/raw_data/labels.csv')
    data = pd.read_csv(path_csv)

    # Initialize an empty list to hold the images
    images = []

    # Iterate over the paths and load each image
    for path in data['pth']:
        raw_data_path = os.path.dirname(os.getcwd())
        raw_data_path = os.path.join(raw_data_path+'/raw_data/'+path)
        images.append(get_image(raw_data_path))

    # Convert the list of images to a numpy array
    X = np.stack(images)

    y = np.array(data.label)

    return X, y

def label_categorize(y):
    '''
    receives y and
    categorizes using to_categorical
    '''
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit the label encoder and transform the labels to numerical labels
    y_encoded = label_encoder.fit_transform(y)

    # Convert numerical labels to one-hot encoded format
    y_cat = to_categorical(y_encoded)
    return y_cat




# PREPROCESSING FOR PREDICTION DATA

def define_faces(image_path):

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ih, iw, _ = image.shape
    # Initialize an empty list to store bounding box coordinates.
    bounding_boxes = []

    # Initialize the face detector.
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # Detect faces in the image.
        results = face_detection.process(image)

    if results.detections:
            for detection in results.detections:
                # Extract the relative bounding box data.
                bboxC = detection.location_data.relative_bounding_box

                # Convert relative coordinates to absolute pixel coordinates
                x_min = int(bboxC.xmin * iw)
                y_min = int(bboxC.ymin * ih)
                width = int(bboxC.width * iw)
                height = int(bboxC.height * ih)

                # Calculate x_max and y_max
                x_max = x_min + width
                y_max = y_min + height

                # Append the bounding box coordinates to the list.
                bounding_boxes.append((x_min, y_min, x_max, y_max))

    return bounding_boxes

def preproc_initial_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def crop_image(image, boxes):

    list_faces = []
    result = []

    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cropped_image = image[y_min:y_max, x_min:x_max]
        list_faces.append(cropped_image)

    for face in list_faces:
        result.append(cv2.resize(face, (96,96)))


    return np.stack(result)

def process_image(image_path):
    # Preprocess the image
    img = preproc_initial_image(image_path)

    # Define faces
    boxes = define_faces(image_path)

    # Crop and resize faces
    faces = crop_image(img, boxes)

    return faces, boxes




#PROCESSING FINAL OUTPUT AFTER EMOTION PREDICTION

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

def image_with_bounding_boxes(image_path, bounding_boxes, texts, output_path):
    """
    Plots an image with bounding boxes and texts, and saves the final image as a JPEG file.

    Parameters:
    - image_path: str, path to the image file
    - bounding_boxes: list of tuples, each containing (x1, y1, x2, y2) coordinates of the bounding box
    - texts: list of str, texts to display on top of each bounding box
    - output_path: str, path to save the output image
    """
    # Load the image
    image = Image.open(image_path)

    # Create a Matplotlib figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
        # Create a rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add text on top of the bounding box
        ax.text(x1, y1 - 10, texts[i], color='red', fontsize=12, backgroundcolor='white')

    # Remove axis for better visualization
    ax.axis('off')

    # Save the final image as a JPEG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='jpeg')

    # Show the plot with bounding boxes and texts
    # plt.show()
