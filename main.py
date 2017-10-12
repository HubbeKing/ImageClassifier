import argparse
import pickle
import os

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np

from multi_gpu import make_parallel
from training import split_data_directory, train_from_directories

# 299x299 images, in RGB format (channels_last)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model", "image_tagger.h5")
INDEX_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model", "class_index.pkl")


def build_model(num_classes):
    """
    @type num_classes: int
    Build an InceptionV3-based image classification model, with num_classes being the number of image classes to classify
    """
    # Base our model on the InceptionV3 convolutional model
    # By using InceptionV3 as a pre-trained base, we can use its feature extractors to "drive" our own classification layer
    base_model = InceptionV3(include_top=False, weights="imagenet")

    # Build a classifier model ontop of the convolutional base layers
    # The final layer has num_classes units, and softmax activation, and thus outputs a list of probabilities, one for each image class
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)

    # Our final model is thus the convolutional parts of InceptionV3 with fully-connected classifier layers ontop
    mod = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model, making it ready for training
    mod.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    # Return the built model, so it can be trained
    return mod


def train_model(mod, training, validation, gpus=0, batch_size=32, epochs=10, save_images=False):
    """
    @type mod: keras.models.Model
    @type training: basestring (directory)
    @type validation: basestring (directory)
    @type gpus: int
    @type batch_size: int
    @type epochs: int
    @type save_images: bool

    Train an InceptionV3-based Keras model's classification layers using the given data directories
    Trains on batches of size 32 for 10 epochs by default, tweakable with the batch_size and epochs parameters
    """
    for layer in mod.layers[:-3]:
        # Freeze all convolutional layers
        layer.trainable = False
    for layer in mod.layers[-3:]:
        # Ensure the fully-connected classifier layers are unfrozen
        layer.trainable = True
    if gpus > 1:
        # If we have specified more than 1 GPU, parallelize with Tensorflow's tf.device
        mod = make_parallel(mod, gpu_count=gpus)
    # Compile model, making it ready for training
    mod.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    # Actually train the model, saving the model after every epoch if model has improved
    # Also save tensorboard-compatible logs for later visualization
    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    tensorboard_log = TensorBoard(log_dir=os.path.join(os.path.dirname(__file__), "save", "logs"))

    image_save_dir = os.path.join(os.path.dirname(__file__), "save", "augmented_images")

    train_from_directories(mod, training, validation,
                           nb_batches=batch_size, nb_epochs=epochs,
                           image_size=(299, 299),
                           callbacks=[checkpointer, tensorboard_log],
                           save_dir=image_save_dir if save_images else False,
                           save_index=True)

    # Save model to file
    mod.save(MODEL_SAVE_PATH)

    return mod


def fine_tune_model(mod, training, validation, gpus=0, batch_size=32, epochs=10):
    """
    @type mod: keras.models.Model
    @type training: basestring (directory)
    @type validation: basestring (directory)
    @type gpus: int
    @type batch_size: int
    @type epochs: int

    Fine-tune an InceptionV3 model's convolutional layers with the given data
    Trains on batches of size 32 for 10 epochs by default, tweakable with the batch_size and epochs parameters
    """
    # TODO double-check which layer to start fine-tuning from, model has 314 layers
    # Currently fine-tuning happens to the top 2 inceptionv3 blocks...ish?
    for layer in mod.layers[:249]:
        layer.trainable = False
    for layer in mod.layers[249:]:
        layer.trainable = True

    if gpus > 1:
        # If we have specified more than 1 GPU, parallelize with Tensorflow's tf.device
        mod = make_parallel(mod, gpu_count=gpus)

    # Compile the model with a tweaked Adam optimizer, with a slow learning rate
    mod.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    # Actually train model, saving the model after every epoch if model has improved
    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    train_from_directories(mod, training, validation,
                           nb_batches=batch_size,
                           nb_epochs=epochs,
                           callbacks=[checkpointer],
                           image_size=(299, 299))

    # Save model and class indices to file
    mod.save(MODEL_SAVE_PATH)

    return mod


def predict_image_classes(mod, classes, image_filepath):
    """
    @type mod: keras.models.Model
    @type classes: dict
    @type image_filepath: basestring

    Given a Keras model, its class index dictionary, and a path to an image file, tries to classify the image using the model
    Prints the classification result as a list of the 5 most likely image classes and the prediction certainty
    """
    img = image.load_img(image_filepath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = mod.predict(x)
    print("{} - Predictions: {}".format(image_filepath, decode_predictions(predictions, classes)))


def decode_predictions(preds, index, top=5):
    """
    Modification of built-in keras ImageNet class decoder, using the image classes found in the training data to decode
    """
    inverse_index = {v: k for k, v in index.items()}
    results = []
    for pred in preds:
        for i in range(len(pred)):
            result = (inverse_index[i], pred[i])  # (class_name, probability)
            results.append(result)
    # Return the top-most probable prediction results
    return sorted(results, key=lambda tup: tup[1], reverse=True)[:top]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A system for classifying images using an InceptionV3-based Convolutional Neural Network")

    # Split args into subcommands
    subparsers = parser.add_subparsers(title="subcommands",
                                       help="Additional help with SUBCOMMAND -h",
                                       dest="command")

    # Training sub-commands
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
                              default=10,
                              help="How large batches to split training data into, defaults to 10")
    train_parser.add_argument("-s", "--save_images",
                              help="Save the augmented images generated during training to save/augmented_images",
                              action="store_true")
    train_parser.add_argument("data_dir",
                              type=str,
                              help="A path to a directory containing the training data")

    # Fine-tuning sub-commands
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
                                  default=10,
                                  help="How large batches to split training data into, defaults to 10")
    fine_tune_parser.add_argument("data_dir",
                                  type=str,
                                  help="A path to a directory containing the training data")

    # Classification sub-commands
    classification_parser = subparsers.add_parser("classify", help="Attempt to classify images")
    classification_parser.add_argument("image_file_path",
                                       type=str,
                                       nargs="+",
                                       help="Images or directories to attempt to classify")

    args = parser.parse_args()
    print(args)

    if args.command == "train":
        # We're training the model, determine how the data dir is set up
        # If it's full of subdirectories, one for each image class, split it 80/20 into training and validation data
        # If it's already been split, just use the training and validation subdirectories as our data source dirs
        if "training" in os.listdir(args.data_dir):
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
            print("Splitting data dir into training and validation data dirs...")
            training_dir, validation_dir = split_data_directory(args.data_dir)
        if not os.path.exists(MODEL_SAVE_PATH):
            # If there's no model saved, build a new one
            model = build_model(len(os.listdir(training_dir)))
        else:
            model = load_model(MODEL_SAVE_PATH)

        # Output a summary of the model
        model.summary()
        print("Total number of layers in model: {}".format(len(model.layers)))

        # Begin training
        trained_model = train_model(model, training_dir, validation_dir,
                                    gpus=args.gpu_count, batch_size=args.batch_size,
                                    epochs=args.epochs, save_images=args.save_images)

        # If the fine_tune flag is set, fine-tune the model after primary training
        if args.fine_tune:
            fine_tune_model(trained_model, training_dir, validation_dir,
                            gpus=args.gpu_count, batch_size=args.batch_size, epochs=args.epochs)

    elif args.command == "fine_tune":
        # Here we can assume the model is at least partially trained, so it'll exist on disk
        # But we do still need to determine how the data dir is set up
        if "training" in os.listdir(args.data_dir):
            if len(os.listdir(os.path.join(args.data_dir, "training"))):
                training_dir = os.path.join(args.data_dir, "training")
                validation_dir = os.path.join(args.data_dir, "validation")
            else:
                # The data dir contains training and validation subdirectories, but they're empty
                # Assume there's more subdirectories in the data dir to gather data from
                training_dir, validation_dir = split_data_directory(args.data_dir)
        else:
            # Data dir is just a collection of subdirectories, one for each image class
            training_dir, validation_dir = split_data_directory(args.data_dir)
        # Now we can just load the model and begin our fine-tuning
        model = load_model(MODEL_SAVE_PATH)
        fine_tune_model(model, training_dir, validation_dir,
                        gpus=args.gpu_count, batch_size=args.batch_size, epochs=args.epochs)

    elif args.command == "classify":
        # If we're just classifying, load the model from disk
        # Then go through the list of given image filepaths and predict the class of each in turn
        model = load_model(MODEL_SAVE_PATH)
        with open(INDEX_SAVE_PATH, "rb") as pickle_file:
            class_index = pickle.load(pickle_file)

        for image_path in args.image_file_path:
            # If image_file_path is a dir, predict on all images inside it
            if os.path.isdir(image_path):
                for img_file in os.listdir(image_path):
                    predict_image_classes(model, class_index, os.path.join(image_path, img_file))
            else:
                predict_image_classes(model, class_index, image_path)
