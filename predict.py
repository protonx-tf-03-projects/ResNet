from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
import pickle

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-file-path", type=str, required=True)
    parser.add_argument("--model-path", default="best_model.h5", type=str)
    parser.add_argument("--class-names-path", default='class_names.pkl', type=str)


   
    args = parser.parse_args()


    print('---------------------Welcome to ResNet-------------------')
    print("Team leader")
    print('Github: dark-kazansky')
    print("Team member")
    print('1. Github: hoangcaobao')
    print('2. Github: sonnymetvn')
    print('3. Github: hoangduc199891')
    print('4. Github: bdghuy')
    print('-------------------------------------------------------- ')
    print('Predict using ResNet model for test file path {0}'.format(args.test_file_path)) # FIXME
    print('===========================')

    # Loading class names
    with open (args.class_names_path, 'rb') as fp:
      class_names = pickle.load(fp)

    # Loading model
    model=load_model(args.model_path)

    # Load test images
    image = preprocessing.image.load_img(args.test_file_path, target_size=(224,224))
    input_arr = preprocessing.image.img_to_array(image)/225
    x = np.expand_dims(input_arr, axis=0)

    predictions = model.predict(x)
    label=np.argmax(predictions)
    print('Result: {}'.format(class_names[label]))