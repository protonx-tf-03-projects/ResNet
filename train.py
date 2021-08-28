from resnet.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from tensorflow import keras
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.data import Dataset
import numpy as np
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.initializers import glorot_uniform
import tensorflow_addons as tfa
import os

if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument('--train-folder', default='Data/Train', type=str, help='Where training data is located')
    parser.add_argument('--valid-folder', default='Data/Validation', type=str, help='Where validation data is located')
    parser.add_argument('--model', default='resnet18', type=str, help='Type of model')
    parser.add_argument('--num-classes', default=1, type=int, help='Number of classes')
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument('--image-size', default=224, type=int, help='Size of input image')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument('--image-channels', default=3, type=int, help='Number channel of input image')
    # parser.add_argument('--model-folder', default='.output/', type=str, help='Folder to save trained model')
    home_dir = os.getcwd()
    args = parser.parse_args()





    # Project Description

    print('---------------------Welcome to ResNet-------------------')
    print('Github: hoangduc199891')
    print('Email: hoangduc199892@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training ResNet model with hyper-params:')
    print('===========================')
    
    # Invoke folder path
    # if args.train_folder != '' and args.valid_folder != '':
    #     # Load train images from folder
    #     train_ds = image_dataset_from_directory(
    #         args.train_folder,
    #         seed=123,
    #         image_size=(args.image_size, args.image_size),
    #         shuffle=True,
    #         batch_size=args.batch_size,
    #     )
    #     val_ds = image_dataset_from_directory(
    #         args.valid_folder,
    #         seed=123,
    #         image_size=(args.image_size, args.image_size),
    #         shuffle=True,
    #         batch_size=args.batch_size,
    #     )

    TRAINING_DIR = args.train_folder
    TEST_DIR = args.valid_folder

    if args.train_folder != '' and args.valid_folder != '':
        training_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(args.image_size, args.image_size), class_mode='binary')
    val_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(args.image_size, args.image_size), class_mode='binary')
    # Create model
    if args.model == 'resnet18':
        model = ResNet18(num_classes = args.num_classes)
    if args.model == 'resnet34':
        model = ResNet34(num_classes = args.num_classes)
    if args.model == 'resnet50':
        model = ResNet50(num_classes = args.num_classes)
    if args.model == 'resnet101':
        model = ResNet101(num_classes = args.num_classes)
    if args.model == 'resnet152':
        model = ResNet152(num_classes = args.num_classes)
    else:
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(args.image_size, args.image_size, args.image_channels)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    model.build(input_shape=(None, args.image_size,
                             args.image_size, args.image_channels))
    optimizer = Adam(learning_rate=args.lr)
    # loss = SparseCategoricalCrossentropy()
    loss = BinaryCrossentropy()
    model.compile(optimizer, loss = loss, metrics=['accuracy'])

    # Traning
    model.fit(
        train_generator,
        steps_per_epoch=8,
        epochs=30,
        verbose=1,
        validation_data=val_generator,
        validation_steps=8)
    # Save model
    # model.save(args.model_folder)