from logic_ml.preprocessing_photo import process_image, decode_predictions, image_with_bounding_boxes
from tensorflow.keras import models
import os

if __name__ == '__main__':

    image_path = 'test.jpg'

    #PREPROCESS INPUT IMAGE
    X_image, boxes = process_image(image_path)

    # GET FER PREDICTIONS
    fer_model = models.load_model('models_trained/ResNet50V2_my_model_VM.h5')

    predictions = fer_model.predict(X_image)

    #UPDATE ORIGINAL PHOTO WITH BOUNDING BOXES AND EMOTION LABEL

    labels = decode_predictions(predictions)

    output_path = 'image_output.jpg'

    image_with_bounding_boxes(image_path, boxes, labels, output_path)
