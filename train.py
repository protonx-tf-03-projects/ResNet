from resnet.model import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, AveragePooling2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os

if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument('--train-folder', default='Data/Train', type=str, help='Where training data is located')
    parser.add_argument('--valid-folder', default='Data/Validation', type=str, help='Where validation data is located')
    parser.add_argument('--model', default='resnet50', type=str, help='Type of model')
    parser.add_argument('--num-classes', default=1, type=int, help='Number of classes')
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--image-size', default=224, type=int, help='Size of input image')
    parser.add_argument('--optimizer', default='adam', type=str, help='Types of optimizers')
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument("--epochs", default=30, type=int, help = 'Number of epochs')
    parser.add_argument("--steps-per-epochs", default=8, type=int, help='Number of steps per epoche')
    parser.add_argument("--validation-step", default=8, type=int, help='Number of validation steps')
    parser.add_argument('--image-channels', default=3, type=int, help='Number channel of input image')
    parser.add_argument('--rotation-range', default=20, type=int, help='Range of image rotation')
    parser.add_argument('--width-shift-range', default=0.2, type=float, help='Range of width shift of image')
    parser.add_argument('--height-shift-range', default=0.2, type=float, help='Range of height shift of image')
    parser.add_argument('--shear-range', default=0.2, type=float, help='Range of shear of image')
    parser.add_argument('--zoom-range', default=0.2, type=float, help='Range of zoom of image')
    parser.add_argument('--horizontal-flip', default=True, type=bool, help='Flip the image horizontally')
    parser.add_argument('--vertical-flip', default=True, type=bool, help='Flip the image vertically')
    parser.add_argument('--validation-split', default=0.2, type=float, help='Ratio of split train to test')


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
    TRAINING_DIR = args.train_folder
    TEST_DIR = args.valid_folder

    if args.train_folder != '' and args.valid_folder != '':
        # Load train images from folder
        training_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=args.rotation_range,
            width_shift_range=args.width_shift_range,
            height_shift_range=args.height_shift_range,
            shear_range=args.shear_range,
            zoom_range=args.zoom_range,
            horizontal_flip=args.horizontal_flip,
            validation_split=args.validation_split)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

    if (args.num_classes > 0 and args.num_classes <= 2):
        loss = BinaryCrossentropy()
        class_mode = 'binary'
        classes = 1
        activation = 'sigmoid'
    else:
        loss = SparseCategoricalCrossentropy()
        class_mode = 'categorical'
        classes = args.num_classes
        activation = 'softmax'

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(args.image_size, args.image_size), class_mode = class_mode)
    val_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(args.image_size, args.image_size), class_mode = class_mode)
    # Create model
    if args.model == 'resnet18':
        model = ResNet18(num_classes = classes, activation=activation)
    if args.model == 'resnet34':
        model = ResNet34(num_classes = classes, activation=activation)
    if args.model == 'resnet50':
        model = ResNet50(num_classes = classes, activation=activation)
    if args.model == 'resnet101':
        model = ResNet101(num_classes = classes, activation=activation)
    if args.model == 'resnet152':
        model = ResNet152(num_classes = classes, activation=activation)
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
        model.add(Dense(classes, activation = activation))

    model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))

    if (args.optimizer == 'adam'):
        optimizer = Adam(learning_rate=args.lr)
    elif (args.optimizer == 'sgd'):
        optimizer = SGD(learning_rate=args.lr)
    elif (args.optimizer == 'rmsprop'):
        optimizer = RMSprop(learning_rate=args.lr)
    elif (args.optimizer == 'adadelta'):
        optimizer = Adadelta(learning_rate=args.lr)
    elif (args.optimizer == 'adamax'):
        optimizer = Adamax(learning_rate=args.lr)
    else:
        raise 'Invalid optimizer. Valid option: adam, sgd, rmsprop, adadelta, adamax'



    model.compile(optimizer, loss = loss, metrics=['accuracy'])

    # Traning
    model.fit(
        train_generator,
        steps_per_epoch=args.steps_per_epochs,
        epochs=args.epochs,
        batch_size = args.batch_size,
        verbose=1,
        validation_data=val_generator,
        validation_steps=args.validation_step)
    # Save model
    # model.save(args.model_folder)