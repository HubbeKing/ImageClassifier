import argparse
import pickle
import os
from six.moves import input

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import load_model, Model
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image
from keras.utils import multi_gpu_model
import numpy as np

from training import split_data_directory, train_from_directories


BASE_DIR = os.path.dirname(__file__)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "image_tagger.h5")
INDEX_SAVE_PATH = os.path.join(BASE_DIR, "model", "class_index.pkl")


class ImageTagger(object):
    """
    An wrapped class around an InceptionResNetV2-based Image Classification model.
    """
    def __init__(self):
        self.model = None
        self.classes = None

    def load_model(self):
        if not os.path.exists(MODEL_SAVE_PATH):
            print("No saved model could be found!")
            return False
        self.model = load_model(MODEL_SAVE_PATH)
        return True

    def load_class_index(self):
        if not os.path.exists(INDEX_SAVE_PATH):
            print("No saved class index could be found!")
            return False
        with open(INDEX_SAVE_PATH, "rb") as pickle_file:
            self.classes = pickle.load(pickle_file)
        return True

    def build_model(self, num_classes):
        # Base the model on the InceptionResNetV2 convolutional model
        # By using InceptionResNetV2 as a pre-trained base,
        # we can use its convolutional layers to provide features for our classification layers to learn from
        base_model = InceptionResNetV2(include_top=False, weights="imagenet")

        # Build a classifier model ontop of the convolutional base layers
        # The final layer has num_classes units, and softmax activation,
        # and thus outputs a list of probabilities, one for each image class
        x = base_model.output
        x = GlobalAveragePooling2D(name="avg_pool")(x)
        x = Dropout(0.2)(x)
        predictions = Dense(num_classes, activation="softmax", name="predictions")(x)

        # Our final model is thus the convolutional parts of InceptionResNetV2
        # with fully-connected classifier layers ontop
        self.model = Model(inputs=base_model.input, outputs=predictions)

        self.model.summary()
        print("Total number of layers in model: {}".format(len(self.model.layers)))

    def save_model(self):
        self.model.save(filepath=MODEL_SAVE_PATH, overwrite=True)

    def train(self, training_dir, validation_dir, gpus=0, batch_size=32, epochs=10, save_images=False):
        """
        @type training_dir: basestring
        @type validation_dir: basestring
        @type gpus: int
        @type batch_size: int
        @type epochs: int
        @type save_images: bool

        Train the classifier model using the given data directories
        """
        # First, freeze the model's convolutional blocks so we only train the classifier layers
        for layer in self.model.layers[:-3]:
            # Freeze all convolutional layers (as they're already trained)
            layer.trainable = False
        for layer in self.model.layers[-3:]:
            # Ensure the fully-connected classifier layers are unfrozen
            layer.trainable = True

        if gpus > 1:
            # If we have specified more than 1 GPU, perform data parallelism
            self.model = multi_gpu_model(self.model, gpus)
            if batch_size // gpus < 32:
                # As each batch will be divided into sub-batches among the GPUs, make sure it's large-ish
                # https://keras.io/utils/#multi_gpu_model
                print("batch_size setting would result in only {} samples per GPU, increasing to {} so each GPU gets 32 samples".format((batch_size // gpus), (32 * gpus)))
                batch_size = 32 * gpus

        # Compile model, making it ready for training
        optimizer = RMSprop(lr=0.045, epsilon=1.0, decay=0.9)  # parameters from https://arxiv.org/pdf/1602.07261.pdf
        self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # Actually train the model, saving the model after every epoch if model has improved
        # Also save tensorboard-compatible logs for later visualization
        checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
        tensorboard_log = TensorBoard(log_dir=os.path.join(BASE_DIR, "save", "logs"))

        image_save_dir = os.path.join(BASE_DIR, "save", "augmented_images")

        # Train the model on the given directories, and save the class index to file
        train_from_directories(self.model, training_dir, validation_dir,
                               batch_size=batch_size,
                               nb_epochs=epochs,
                               image_size=(299, 299),
                               callbacks=[checkpointer, tensorboard_log],
                               save_dir=image_save_dir if save_images else False,
                               save_index=True)

        # Save model to file
        self.save_model()

    def fine_tune(self, training_dir, validation_dir, gpus=0, batch_size=32, epochs=10):
        """
        @type training_dir: basestring
        @type validation_dir: basestring
        @type gpus: int
        @type batch_size: int
        @type epochs: int

        Fine-tune the classifier model's top convolutional layers with the given data directories
        """

        # TODO check these, as they are valid for InceptionV3, but InceptionResNetV2 has different architecture
        for layer in self.model.layers[:249]:
            layer.trainable = False
        for layer in self.model.layers[249:]:
            layer.trainable = True

        if gpus > 1:
            # If we have specified more than 1 GPU, perform data parallelism
            self.model = multi_gpu_model(self.model, gpus)
            if batch_size // gpus < 32:
                # As each batch will be divided into sub-batches among the GPUs, make sure it's large-ish
                # https://keras.io/utils/#multi_gpu_model
                print("batch_size setting would result in only {} samples per GPU, increasing to {} so each GPU gets 32 samples".format((batch_size // gpus), (32 * gpus)))
                batch_size = 32 * gpus

        # Compile the model with a tweaked SGD optimizer with a slow learning rate
        # This make sure the updates done to the weights stays small, so we don't break things
        self.model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])

        # Actually train model, saving the model after every epoch if model has improved
        checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
        train_from_directories(self.model, training_dir, validation_dir,
                               batch_size=batch_size,
                               nb_epochs=epochs,
                               callbacks=[checkpointer],
                               image_size=(299, 299))

        # Save model to file
        self.save_model()

    def classify_image(self, image_filepath):
        if self.classes is None:
            with open(INDEX_SAVE_PATH, "rb") as pickle_file:
                self.classes = pickle.load(pickle_file)

        img = image.load_img(image_filepath, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predictions = self.model.predict(x, batch_size=1)
        print("{} - Predictions: {}".format(image_filepath, self.decode_predictions(predictions)))

    def decode_predictions(self, preds, top=5):
        """
        Modification of Keras' built-in ImageNet class decoder (keras.applications.imagenet_utils.decode_predictions)
        Uses the image classes found in the training data to decode a model prediction
        """
        inverse_index = {v: k for k, v in self.classes.iteritems()}
        results = []
        for pred in preds:
            for i in range(len(pred)):
                result = (inverse_index[i], pred[i])  # (class_name, probability)
                results.append(result)
        # return the top-most probable image classes
        return sorted(results, key=lambda tup: tup[1], reverse=True)[:top]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A system for classifying images using a Convolutional Neural Network")

    # Split args into subcommands
    subparsers = parser.add_subparsers(title="subcommands",
                                       help="Additional help with SUBCOMMAND -h",
                                       dest="command")
    # Build sub-command
    build_parser = subparsers.add_parser("build",
                                         help="Build a convolutional model")
    build_parser.add_argument("data_dir",
                              type=str,
                              help="A path to a directory containing the data the model is going to be trained on")

    # Train sub-command
    train_parser = subparsers.add_parser("train",
                                         help="Train the model's classification layers")
    train_parser.add_argument("-f", "--fine_tune",
                              help="Do additional training after main training to fine-tune the convolutional layers of the model",
                              action="store_true")
    train_parser.add_argument("-g", "--gpu_count",
                              type=int,
                              default=0,
                              help="How many GPUs to use when training the model, only needed when >1 GPUs")
    train_parser.add_argument("-e", "--epochs",
                              type=int,
                              default=10,
                              help="How many epochs to train for, defaults to 10")
    train_parser.add_argument("-b", "--batch_size",
                              type=int,
                              default=32,
                              help="How large batches to split training data into, defaults to 32")
    train_parser.add_argument("-s", "--save_images",
                              help="Save the augmented images generated during training to save/augmented_images",
                              action="store_true")
    train_parser.add_argument("data_dir",
                              type=str,
                              help="A path to a directory containing the training data")

    # Fine-tune sub-command
    fine_tune_parser = subparsers.add_parser("fine_tune",
                                             help="Fine-tune the model's convolutional layers")
    fine_tune_parser.add_argument("-g", "--gpu_count",
                                  type=int,
                                  default=0,
                                  help="How many GPUs to use when fine-tuning the model, only needed when >1 GPUs")
    fine_tune_parser.add_argument("-e", "--epochs",
                                  type=int,
                                  default=10,
                                  help="How many epochs to train for, defaults to 10")
    fine_tune_parser.add_argument("-b", "--batch_size",
                                  type=int,
                                  default=32,
                                  help="How large batches to split training data into, defaults to 32")
    fine_tune_parser.add_argument("data_dir",
                                  type=str,
                                  help="A path to a directory containing the training data")

    # Classify sub-command
    classification_parser = subparsers.add_parser("classify", help="Attempt to classify images")
    classification_parser.add_argument("image_file_path",
                                       type=str,
                                       nargs="+",
                                       help="Images or directories to attempt to classify")

    args = parser.parse_args()
    # print(args)

    if args.command == "classify":
        # If we're just classifying, load the model from disk
        # Then go through the list of given image filepaths and predict the class of each in turn
        classifier = ImageTagger()
        model_loaded = classifier.load_model()
        index_loaded = classifier.load_class_index()

        if not model_loaded and not index_loaded:
            print("Model and class index could not be loaded.")
        elif not model_loaded:
            print("Model could not be loaded.")
        elif not index_loaded:
            print("Class index could not be loaded.")
        else:
            for image_path in args.image_file_path:
                # If image_file_path is a dir, predict on all images inside it
                if os.path.isdir(image_path):
                    for img_file in os.listdir(image_path):
                        classifier.classify_image(os.path.join(image_path, img_file))
                else:
                    classifier.classify_image(image_path)

    # Next, determine how the data dir is set up
    # If it's full of subdirectories, one for each image class, split it 70/30 into training and validation data
    # If it's already been split, just use the training and validation subdirectories as our data source dirs
    if hasattr(args, "data_dir") and "training" in os.listdir(args.data_dir):
        if len(os.listdir(os.path.join(args.data_dir, "training"))):
            training_dir = os.path.join(args.data_dir, "training")
            validation_dir = os.path.join(args.data_dir, "validation")
        else:
            # The data dir contains training and validation subdirectories, but they're empty
            # Assume there's more subdirectories in the data dir to gather data from
            print("Splitting data dir into training and validation data dirs...")
            training_dir, validation_dir = split_data_directory(args.data_dir)
    else:
        # Data dir is just a collection of subdirectories, one for each image class
        # Split it into training and validation data
        print("Splitting data dir into training and validation data dirs...")
        training_dir, validation_dir = split_data_directory(args.data_dir)

    if args.command == "build":
        overwrite = True
        if os.path.exists(MODEL_SAVE_PATH):
            print("A saved model already exists!")
            overwrite = input("Overwrite saved model? (Y/N) ")
            if overwrite.upper() != "Y":
                overwrite = False
        if not os.path.exists(MODEL_SAVE_PATH) or overwrite:
            classifier = ImageTagger()
            classifier.build_model(len(os.listdir(training_dir)))
            classifier.save_model()

            print("Successfully built model and saved to {}".format(MODEL_SAVE_PATH))

    elif args.command == "train":
        # Load the model from file
        classifier = ImageTagger()
        classifier.load_model()

        # Train the model
        classifier.train(training_dir, validation_dir,
                         gpus=args.gpu_count, batch_size=args.batch_size,
                         epochs=args.epochs, save_images=args.save_images)

        # If the fine_tune flag is set, fine-tune the model after primary training
        if args.fine_tune:
            classifier.fine_tune(training_dir, validation_dir,
                                 gpus=args.gpu_count,
                                 batch_size=args.batch_size,
                                 epochs=args.epochs)

    elif args.command == "fine_tune":
        # Simply load the model and begin fine-tuning
        classifier = ImageTagger()
        classifier.load_model()

        classifier.fine_tune(training_dir, validation_dir,
                             gpus=args.gpu_count,
                             batch_size=args.batch_size,
                             epochs=args.epochs)
