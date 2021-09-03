import os
import tensorflow as tf
from model import *
import numpy as np
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.models import *
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-image", default="test.png", type=str, required=True)
    parser.add_argument("--model-path", default="best_model.h5", type=str)

   
    args = parser.parse_args()


    print('---------------------Welcome to ResNet-------------------')
    print("Team leader")
    print('Github: dark-kazansky')
    print("Team member")
    print('1. Github: hoangcaobao')
    print('2. Github: sonnymetvn')
    print('3. Github: hoangduc199891')
    print('4. Github: bdghuy')
    print('---------------------------------------------------------------------')
    print('Predict using ResNet model for test file path {0}'.format(args.test_file_path)) # FIXME
    print('===========================')

    print("===================LOADING MODEL==========================")
    model=load_model(args.model_path)
    print("================FINISH LOADING MODE=======================")

    print("===================LOADING TEST IMAGE=====================")
    image=load_img(args.test_image)
    input_arr=img_to_array(image)
    input_arr=np.array([input_arr])
    print("================FINISH LOADING TEST IMAGE=================")

    print("======================PREDICTING==========================")
    result=np.argmax(model.predict(input_arr),axis=1)
    print("YOUR RESULT AFTER GO THROUGH RESNET {}".format(result))
    print("====================END PREDICTING========================")
