# Facial Emotion Recognition
The following repository is a real-time face detection and emotion classification model.
<p align="center">
    <img src="output.gif">
</p>



The face detection is powered by [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide).

The emotion classification model is a built on an CNN architecture called Inception-Resnet-V2 with weights retrained on a subsample of the Affectnet Dataset.

## The dataset: AffectNet
The dataset used to train the model was a subsample of the AffectNet dataset downloaded from Kaggle at this [link](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data).

<p align="center">
    <img src="https://www.catalyzex.com/_next/image?url=https%3A%2F%2Fai2-s2-public.s3.amazonaws.com%2Ffigures%2F2017-08-08%2Fcb243d093ecd339eda05205b7b2035a4b66a63f3%2F1-Figure1-1.png&w=640&q=75">
</p>

It comprises 28 thousand images labeled into 8 different emotions. The dataset is balanced and images are of size 96x96.


## The emotion classification model: Inception-Resnet-V2
The best results stemmed from the Inception-Resnet-V2 architecture with an accuracy of 70% on the test set. The model architecture was imported from the [Keras library](https://keras.io/api/applications/inceptionresnetv2/).

Prior to this model, we trained a customized CNN model (40% accuracy), and a model with the ResNet50V2 architecture (63% accuracy).


<p align="center">
    <img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*6rGFjtxxqhDbIsfC9buEhA.jpeg">
</p>




## Get to see use our code :)
* Clone this commit to your local machine using

```bash
git clone git@github.com:JeanLucaSchindler/FER.git
```

* Install these dependencies with pip install

```bash
pip install -r ../REQUIREMENTS.txt
```

* Download pretrained model `model_chelou.h5` from [here](https://drive.google.com/file/d/1FbD9i83YNRpXwvbLhwROjR7OXkjTT7MZ/view?usp=drive_link).

* Place `model_chelou.h5` into `../models_trained/`.
