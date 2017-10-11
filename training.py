import os
from math import ceil
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
            # Discard them if so, so we can delete the folder
            os.remove(os.path.join(data_dir, image_class, f))
        os.rmdir(os.path.join(data_dir, image_class))

    return os.path.join(data_dir, "training"), os.path.join(data_dir, "validation")


def train_from_directories(model, training_dir, validation_dir, nb_batches=32, nb_epochs=50, image_size=None, save_dir=None, callbacks=None):
    """
    @type model: keras.models.Model
    @type training_dir: basestring
    @type validation_dir: basestring
    @type nb_batches: int
    @type nb_epochs: int
    @type image_size: None | tuple (int, int)
    @type save_dir: None | basestring
    @type callbacks: None | list

    Trains the given Keras model using the given paths as data sources for .flow_from_directory
    Counts the number of images in the dataset to ensure each training epoch covers the entire dataset
    Trains with a batch size of 32, for 50 epochs by default, changeable by altering the nb_batches and nb_epoch parameters
    If save_dir is a string, it's treated as a path to save transformed input data to (useful for visualizing preprocessing results)
    Callsbacks may be a list of keras callbacks to supply to fit_generator
    Returns a dictionary mapping class names to class indices, for use when using model to predict data
    """

    # The number of steps per epoch should be equal to the number of unique images divided by the batch size
    # This means one epoch is one full pass-through of the data

    training_steps = ceil((sum([len(files) for r, d, files in os.walk(training_dir)]) / float(nb_batches)))
    validation_steps = ceil((sum([len(files) for r, d, files in os.walk(validation_dir)]) / float(nb_batches)))

    from main import DATA_FORMAT

    # Perform real-time data augmentation to (hopefully) get better end-results
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1./255,
        data_format=DATA_FORMAT
    )

    if image_size is None:
        # Get the image size defined in main.py
        from main import IMAGE_FORMAT
        image_size = (IMAGE_FORMAT[0], IMAGE_FORMAT[1])

    train_generator = datagen.flow_from_directory(
        directory=training_dir,
        target_size=image_size,
        batch_size=nb_batches,
        class_mode="categorical",
        save_to_dir=save_dir,
        save_prefix="sample"
    )

    from main import INDEX_SAVE_PATH
    with open(INDEX_SAVE_PATH, "wb") as pickle_file:
        pickle.dump(train_generator.class_indices, pickle_file, pickle.HIGHEST_PROTOCOL)

    validation_generator = datagen.flow_from_directory(
        directory=validation_dir,
        target_size=image_size,
        batch_size=nb_batches,
        class_mode="categorical"
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=training_steps,
        epochs=nb_epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    return train_generator.class_indices
