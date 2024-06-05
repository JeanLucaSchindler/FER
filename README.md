# FER
Our goal: Making computers more empathetic than humans...

## First step - Preprocessing and normalizing our data

The data that would be fed into the network is labeled as:
- train_images with a shape of **(4743, 128, 128, 3)** representing **4743** tuples of size **96 x 96 pixels** (width and height) in 3 dimensions (RGB).
- test_images with a shape of **(587, 128, 128, 3)**
- validation_images with a shape of **(526, 128, 128, 3)**
- train_labels with a shape of **(4743,)** representing an array of 0s or 1s with 4743 entries. A shape of **(4743,1)** with each label coded in a separate row would also work.
- test_labels with a shape of **(587,)**
- validation_labels with a shape of **(526,)**


## Second step - Baseline model
Training a basic CNN that served as our baseline, and whose result was 0.40 of accuracy.
It was constructed with the following layers:
- Conv2D(64)
- AvgMaxPool(4,4)
- Conv2D(32)
- AvgMaxPool(2,2)
- Conv2D(16)
- Flatten()
- Dense(64)
- Dropout(0.2)
- Dense(8)


## Third step - Transfer Learning
Select 4 pre-trained models that could help in fulfilling our task. These are the following:
- ResNet50V2
- Inception-ResNet
- Inception V3
- VGG19


### Why ResNet?
--> one of the most widely used and effective deep learning models to date
--> Architecture design is inspired on VGG-19 and has a 34-layer plain network to which shortcut and skip connections are added.
--> uses **a lot of batch-normalization** to let it train hundreds of layers successfully without sacrificing speed over time.
  --> also why we added a layer of BatchNormalization() in our model

#### Why ResNet50V2?
--> depth of 103 and 50 trainable layers, of which 50 have tunable weights (48 convolution layers, 1 maximum pooling, and 1 average pooling layer).
--> residual network because it is known to overcome the “vanishing gradient problem.”, by using “skip connections” which allow the gradient to flow unhindered, making it possible to construct a network of many deep layers of convolutional layers with efficiency.


### Why Inception?
--> utilised for a variety of computer vision applications, including face detection and identification, adversarial training.
  --> made up of nine inception modules stacked on top of each other, with **max-pooling layers** between them.
--> all layers are followed by a batchnorm and ReLU nonlinearity.

#### Why Inception V3?
--> lowering the computational cost of deep networks without sacrificing generalisation.
--> Inception-V3 employs a 1x1 convolutional operation, which divides the input data into three or four smaller 3D spaces, and then maps all correlations in these smaller 3D spaces using standard (3x3 or 5x5) convolutions.

### Why Inception-ResNet?
--> lowering the computational cost of deep networks without sacrificing generalisation.
--> nception-V4 with residual connections (Inception-ResNet) has the same generalisation capacity as plain InceptionV4, but with more depth and width. They did notice, however, that Inception-ResNet converges faster than Inception-V4, indicating that training Inception networks with residual connections speeds up the process dramatically.

### Why VGG?
--> defined by its simplicity, with simply pooling layers and a fully linked layer as additional components.
--> 19 layers deep to replicate the relationship between depth and network representational capability.
--> Small 3 x 3 filters are used in the network.


"""based upon article : https://iq.opengenus.org/different-types-of-cnn-models/"""


## Fourth step - Initialize models and start fine-tuning them

**taken from text - to be modified**
- input shape = (128, 128, 3):The original model starts with an input image size of 224 x 224 x 3. Our image size is 128 x 128 x 3. We need to change the image shape of the input layer so we can use the model on our images. This parameter should have exactly 3 inputs channels, and width and height should be no smaller than 32.
- weights = 'imagenet'If we want to use the weights of the original pre-trained model, we need to set them to ‘imagenet’. This way the model will use its prior knowledge about the visual features it detected while getting trained on ImageNet dataset. Using pre-trained weights is also beneficial as it helps the model converge in less epochs.
- include_top = False:The original model is built to classify 1000 categories as you can see in the final output layer (1000 neurons). We need to remove last layer from the model and include our own dense layer with a single neuron that would be used to make a binary classification decision — 0 (normal) or 1 (pneumonia). We need to remove the top layers by setting include_top to False and by add a final fully connected layer with one neuron and an activation function of sigmoid (which is used for binary classification)model.add(Dense(1, activation = "sigmoid"))
- pooling = "avg"This pooling mode for feature extraction is meaningful only when include_top is False. Removing the top will also remove the pooling layer prior to the final classification layer, and so this step is needed to re-shape the data to make it suitable as input for the final classification layer. Unlike Flatten(), global pooling condenses all of the feature maps and relevant information into a single multi dimensional one in a way a single dense classification layer can understand and classify directly from there. It’s typically applied as average pooling (GlobalAveragePooling2D) or max pooling (GlobalMaxPooling2D). In this project we will use average pooling as in the original model by specifying pooling = "avg".
- model.add(Flatten())Another option would be to flatten the output layer from the CNN rather than applying global pooling. Flattening simply converts the multi-dimensional feature map into a 1-dimensional long feature vector for inputting it to the next layer. However, we would need to train one or more dense (fully-connected) layers afterwards to discern patterns from this long vector before reaching the final classification layer. Also,flatten will result in a larger dense layer afterwards, which might be more expensive and may result in worse overfitting.
- base_model.trainable = False Finally during training, we will freeze the base_model layers, these layers won’t be trainable and the weights of those layers won’t be updated. This way we avoid destroying any of the information the model contains during future training rounds, and this way higher accuracy can be achieved even for smaller datasets. We will be adding a few new, trainable layers on top of the frozen layers. They will learn to turn the old features into predictions on a new dataset.

Result: the two best models in terms of both accuracy/val_accuracy and the threshold between loss and accuracy were: 1) ResNet50V2; 2) Inception-ResNet.
