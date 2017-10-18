import argparse
import pickle
import os

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPooling2D
from keras.models import load_model, Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
import numpy as np

from multi_gpu import make_parallel
from training import split_data_directory, train_from_directories


BASE_DIR = os.path.dirname(__file__)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "model", "image_tagger.h5")
INDEX_SAVE_PATH = os.path.join(BASE_DIR, "model", "class_index.pkl")


def build_custom_model(num_classes):
    """
    @type num_classes: int

    Build a custom convolutional model for image classification,
    with num_classes being the number of different image classes
    """
    mod = Sequential()

    # First, add 3 convolutional blocks
    mod.add(Conv2D(64, (3, 3), input_shape=(299, 299, 3)))
    mod.add(Activation("relu"))
    mod.add(MaxPooling2D(pool_size=(2, 2)))

    mod.add(Conv2D(64, (3, 3)))
    mod.add(Activation("relu"))
    mod.add(MaxPooling2D(pool_size=(2, 2)))

    mod.add(Conv2D(128, (3, 3)))
    mod.add(Activation("relu"))
    mod.add(MaxPooling2D(pool_size=(2, 2)))
    # The model now generates feature maps

    # Flatten these down to 1D feature tensors and classify
    mod.add(Flatten())
    mod.add(Dense(128))
    mod.add(Activation("relu"))
    # A rather aggressive dropout should help prevent overfitting on small datasets
    mod.add(Dropout(0.5))
    mod.add(Dense(num_classes))
    mod.add(Activation("softmax"))

    return mod


def build_inceptionv3_model(num_classes):
    """
    @type num_classes: int
    Build an InceptionV3-based image classification model
    with num_classes being the number of different image classes
    """
    # Base our model on the InceptionV3 convolutional model
    # By using InceptionV3 as a pre-trained base,
    # we can use its feature extractors to "drive" our own classification layers
    base_model = InceptionV3(include_top=False, weights="imagenet")

    # Build a classifier model ontop of the convolutional base layers
    # The final layer has num_classes units, and softmax activation,
    # and thus outputs a list of probabilities, one for each image class
    x = base_model.output
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax", name="predictions")(x)

    # Our final model is thus the convolutional parts of InceptionV3
    # with fully-connected classifier layers ontop
    return Model(inputs=base_model.input, outputs=predictions)


def train_model(mod, training, validation, gpus=0, batch_size=32, epochs=10, save_images=False):
    """
    @type mod: keras.models.Model
    @type training: basestring (directory)
    @type validation: basestring (directory)
    @type gpus: int
    @type batch_size: int
    @type epochs: int
    @type save_images: bool

    Train the given model using the given directories
    If the model is InceptionV3-based
    freeze the convolutional blocks and only train the top classification layers
    """
    if len(mod.layers) > 20:  # This is an inceptionV3-based model
        # Freeze the model's convolutional blocks so we only train the classifier layers
        for layer in mod.layers[:-3]:
            # Freeze all convolutional layers (as they're already trained)
            layer.trainable = False
        for layer in mod.layers[-3:]:
            # Ensure the fully-connected classifier layers are unfrozen
            layer.trainable = True
    if gpus > 1:
        # If we have specified more than 1 GPU, parallelize with Tensorflow's tf.device
        mod = make_parallel(mod, gpu_count=gpus)
    # Compile model, making it ready for training
    mod.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    # Actually train the model, saving the model after every epoch if model has improved
    # Also save tensorboard-compatible logs for later visualization
    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    tensorboard_log = TensorBoard(log_dir=os.path.join(BASE_DIR, "save", "logs"))

    image_save_dir = os.path.join(BASE_DIR, "save", "augmented_images")

    train_from_directories(mod, training, validation,
                           batch_size=batch_size,
                           nb_epochs=epochs,
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

    Fine-tune a model's top convolutional layers with the given data
    """
    if len(mod.layers) > 20:  # This is an inceptionV3-based model
        # Freeze the bottom of the model,
        # ensure the top 2 convolutional blocks and the classifier layers are unfrozen
        # TODO check actual model architecture and make sure we're fine-tuning on the things we want to
        for layer in mod.layers[:249]:
            layer.trainable = False
        for layer in mod.layers[249:]:
            layer.trainable = True
    else:  # This is a custom model
        # Freeze the bottom-most convolutional block,
        # ensure the remainder of the model is unfrozen
        for layer in mod.layers[:3]:
            layer.trainable = False
        for layer in mod.layers[3:]:
            layer.trainable = True

    if gpus > 1:
        # If we have specified more than 1 GPU, parallelize with Tensorflow's tf.device
        mod = make_parallel(mod, gpu_count=gpus)

    # Compile the model with a tweaked Adam optimizer, with a slow learning rate
    mod.compile(optimizer=Adam(lr=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

    # Actually train model, saving the model after every epoch if model has improved
    checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_best_only=True, verbose=1)
    train_from_directories(mod, training, validation,
                           batch_size=batch_size,
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

    Uses the given model to predict the image class for an image
    Prints the top 5 predictions for the image
    """
    img = image.load_img(image_filepath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    predictions = mod.predict(x, batch_size=1)
    print("{} - Predictions: {}".format(image_filepath, decode_predictions(predictions, classes)))


def decode_predictions(preds, index, top=5):
    """
    Modification of built-in keras ImageNet class decoder,
    using the image classes found in the training data to decode
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
    parser = argparse.ArgumentParser(description="A system for classifying images using a Convolutional Neural Network")

    # Split args into subcommands
    subparsers = parser.add_subparsers(title="subcommands",
                                       help="Additional help with SUBCOMMAND -h",
                                       dest="command")
    # Building sub-command
    build_parser = subparsers.add_parser("build",
                                         help="Build a convolutional model")
    build_parser.add_argument("type",
                              type=str,
                              help="The model type to build, either \"inception_v3\" for a tweaked InceptionV3-model or \"custom\" for a custom model")
    build_parser.add_argument("data_dir",
                              type=str,
                              help="A path to a directory containing the data the model is going to be trained on")

    # Training sub-command
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

    # Fine-tuning sub-command
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

    if args.command == "classify":
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

    # First, determine how the data dir is set up
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
            overwrite = raw_input("Overwrite saved model? (Y/N)")
            if overwrite.upper() != "Y":
                overwrite = False
        if not os.path.exists(MODEL_SAVE_PATH) or overwrite:
            if args.type not in ["inception_v3", "custom"]:
                print("Unknown model type - must be either \"inception_v3\" or \"custom\"")
            if args.type == "inception_v3":
                model = build_inceptionv3_model(len(os.listdir(training_dir)))
            else:
                model = build_custom_model(len(os.listdir(training_dir)))
            model.save(filepath=MODEL_SAVE_PATH, overwrite=True)
            print("Successfully built model and saved to {}".format(MODEL_SAVE_PATH))

    elif args.command == "train":
        # Load the model from file
        model = load_model(MODEL_SAVE_PATH)

        # Output a summary of the model
        model.summary()
        print("Total number of layers in model: {}".format(len(model.layers)))

        # Train the model
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
