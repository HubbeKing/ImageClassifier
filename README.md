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

* `git clone https://github.com/HubbeKing/ImageClassifier.git`
* `cd ImageClassifier`
* `virtualenv env`
* `source env/bin/activate`
* `pip install -r requirements.txt`
* Optionally, for GPU support `pip install tensorflow-gpu`

### Windows Installation

* Install in a Docker container using the provided Dockerfile
* Mount input data as a volume during training, or modify Dockerfile to add image data to desired location

* `git clone https://github.com/HubbeKing/ImageClassifier.git`
* `cd ImageClassifier`
* `docker build -t classifier .`

### Usage

* See `python main.py -h` on linux OR `docker run classifier` on Windows
