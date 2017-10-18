import os
import pickle
import random
from shutil import move

from keras.preprocessing.image import ImageDataGenerator


def split_data_directory(data_dir, training_size=0.70):
    """
    Splits a given data directory into training and validation subdirectories
    Randomly picks 70% of data per subdirectory (image class) to training set and 30% to validation set
    """
    image_classes = os.listdir(data_dir)
    os.mkdir(os.path.join(data_dir, "training"))
    os.mkdir(os.path.join(data_dir, "validation"))
    for image_class in image_classes:
        training_dir = os.path.join(data_dir, "training", image_class)
        validation_dir = os.path.join(data_dir, "validation", image_class)
        os.mkdir(training_dir)
        os.mkdir(validation_dir)
        filelist = os.listdir(os.path.join(data_dir, image_class))
        random.shuffle(filelist)
        for source_file in filelist[:int(len(filelist) * training_size)]:
            move(os.path.join(data_dir, image_class, source_file), training_dir)
        for source_file in filelist[int(len(filelist) * training_size):]:
            try:
                move(os.path.join(data_dir, image_class, source_file), validation_dir)
            except (IOError, OSError):
                # There could be slight overlap, since len(os.listdir(data_dir)) can be an arbitrary number
                continue

        for f in os.listdir(os.path.join(data_dir, image_class)):
            # If the source folder contains an odd number of files, we might miss one or three
            # Move them to the training folder, so that the source is empty and can be removed
            move(os.path.join(data_dir, image_class, f), training_dir)
        os.rmdir(os.path.join(data_dir, image_class))

    return os.path.join(data_dir, "training"), os.path.join(data_dir, "validation")


def train_from_directories(model, training_dir, validation_dir, image_size, batch_size=32, nb_epochs=50, save_dir=None, callbacks=None, save_index=False):
    """
    @type model: keras.models.Model
    @type training_dir: basestring
    @type validation_dir: basestring
    @type image_size: tuple (int, int)
    @type batch_size: int
    @type nb_epochs: int
    @type save_dir: None | basestring
    @type callbacks: None | list
    @type save_index: bool

    Trains the given Keras model using the given paths as data sources for .flow_from_directory
    Counts the number of images in the dataset to ensure each training epoch covers the entire dataset
    Trains with a batch size of 32, for 50 epochs by default, changeable by altering the nb_batches and nb_epoch parameters
    If save_dir is a string, it's treated as a path to save transformed input data to (useful for visualizing preprocessing results)
    Callsbacks may be a list of keras callbacks to supply to fit_generator
    Returns a dictionary mapping class names to class indices, for use when using model to predict data
    Saves the class index to file if save_index=True
    """

    # Perform real-time data augmentation to (hopefully) get better end-results
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        directory=training_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        save_to_dir=save_dir,
        save_prefix="sample"
    )
    if save_index:
        from main import INDEX_SAVE_PATH
        with open(INDEX_SAVE_PATH, "wb") as pickle_file:
            pickle.dump(train_generator.class_indices, pickle_file)

    validation_generator = test_datagen.flow_from_directory(
        directory=validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical"
    )

    # Train the model using the created generators
    # steps_per_epoch will be equal to the number of samples divided by the batch size,
    # so each epoch should cover the entire dataset once
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=nb_epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=callbacks
    )
