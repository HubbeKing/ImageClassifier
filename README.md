# ImageClassifier

A tool for training and using an Inception-Resnet-V2 convolutional network for image classification, using transfer learning.

Originally created as part of a Masters Thesis project in Computer Science.

### License information

* This project uses the Inception model weights provided by Keras
* These are provided under the Apache 2.0 license
* https://github.com/tensorflow/models/blob/master/LICENSE
* Consequently, this work is also licensed under the Apache 2.0 license

### Linux Installation

* Known to work with python versions 2.7 and 3.6, should work with 3.7 once Tensorflow is available for it
* Using a virtualenv is highly recommended
* For GPU training, tensorflow-gpu is required

```sh
git clone https://github.com/HubbeKing/ImageClassifier.git`
cd ImageClassifier
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
Optionally, for GPU support `pip install tensorflow-gpu`

### Windows Installation

* Install in a Docker container using the provided Dockerfile
* Mount input data as a volume during training, or modify Dockerfile to add image data to desired location

```sh
git clone https://github.com/HubbeKing/ImageClassifier.git`
cd ImageClassifier
docker build -t classifier .
```

### Usage details

* See `python main.py -h` on linux OR `docker run classifier` on Windows for specifics and more arguments

### Example usage
```sh
# Build an untrained model based on your input data
python main.py build path/to/training_data

# Train the model for a number of epochs on your input data
python main.py train --epochs=50 path/to/training_data

# Test out the model to see how it performs on an image or folder containing images
python main.py classify path/to/image.jpg path/to/other_image.jpg path/to/folder_of_images

# If needed, fine-tune the model for greater performance
python main.py fine_tune --epochs=50 path/to/training_data

# Test model again to see improvements
python main.py classify path/to/image.jpg path/to/other_image.jpg path/to/folder_of_images
```
