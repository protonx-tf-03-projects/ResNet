from model import resnet18, resnet34, resnet50, resnet101, resnet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D,BatchNormalization, AveragePooling2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adamax
from argparse import ArgumentParser
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import os

if __name__ == "__main__":
    parser = ArgumentParser()

    # Arguments users used when running command lines
    parser.add_argument('--train-folder', default='Data/Train', type=str, help='Where training data is located')
    parser.add_argument('--valid-folder', default='Data/Validation', type=str, help='Where validation data is located')
    parser.add_argument('--model', default='resnet50', type=str, help='Type of model')
    parser.add_argument('--num-classes', default=2, type=int, help='Number of classes')
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument('--image-size', default=224, type=int, help='Size of input image')
    parser.add_argument('--optimizer', default='adam', type=str, help='Types of optimizers')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=120, type=int, help = 'Number of epochs')
    parser.add_argument('--image-channels', default=3, type=int, help='Number channel of input image')
    parser.add_argument('--class-mode', default='sparse', type=str, help='Class mode to compile')
    parser.add_argument('--model-path', default='best_model.h5', type=str, help='Path to save trained model')
    parser.add_argument('--class-names-path', default='class_names.pkl', type=str, help='Path to save class names')


    # parser.add_argument('--model-folder', default='.output/', type=str, help='Folder to save trained model')
    args = parser.parse_args()

    # Project Description

    print('---------------------Welcome to resnet-------------------')
    print('Github: hoangduc199891')
    print('Email: hoangduc199892@gmail.com')
    print('---------------------------------------------------------------------')
    print('Training resnet model with hyper-params:')
    print('===========================')

    # Invoke folder path
    TRAINING_DIR = args.train_folder
    TEST_DIR = args.valid_folder
    
    loss = SparseCategoricalCrossentropy()
    class_mode = args.class_mode
    classes = args.num_classes
        
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = training_datagen.flow_from_directory(TRAINING_DIR, target_size=(224, 244), batch_size= 64, class_mode = class_mode )
    val_generator = val_datagen.flow_from_directory(TEST_DIR, target_size=(224, 224), batch_size= 64, class_mode = class_mode)

    class_names=list(train_generator.class_indices.keys())
    with open(args.class_names_path,'wb') as fp:
      pickle.dump(class_names, fp)

    # Create model
    if args.model == 'resnet18':
        model = resnet18(num_classes = classes)
    elif args.model == 'resnet34':
        model = resnet34(num_classes = classes)
    elif args.model == 'resnet50':
        model = resnet50(num_classes = classes)
    elif args.model == 'resnet101':
        model = resnet101(num_classes = classes)
    elif args.model == 'resnet152':
        model = resnet152(num_classes = classes)
    else:
        print('Wrong resnet name, please choose one of these model: resnet18, resnet34, resnet50, resnet101, resnet152')

    model.build(input_shape=(None, args.image_size, args.image_size, args.image_channels))
    model.summary()


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



    model.compile(optimizer=optimizer, 
                loss=SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    
    best_model = ModelCheckpoint(args.model_path,
                                 save_weights_only=False,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 mode='max',
                                 save_best_only=True)
    # Traning
    model.fit(
        train_generator,
        epochs=args.epochs,
        verbose=1,
        validation_data=val_generator,
        callbacks=[best_model])