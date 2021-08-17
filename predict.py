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
    parser.add_argument("--model-folder", default="model", type=str)

   
    args = parser.parse_args()


    print('---------------------Welcome to ResNet-------------------')
    print('Github: hoangcaobao')
    print('Email: caobaohoang03@gmail.com')
    print('---------------------------------------------------------------------')
    print('Predict using ResNet model for test file path {0}'.format(args.test_file_path)) # FIXME
    print('===========================')

    print("===================LOADING MODEL==========================")
    model=load_model(args.model_folder)
    print("================FINISH LOADING MODE=======================")

    print("===================LOADING TEST IMAGE=====================")
    image=load_img(args.test_image)
    input_arr=img_to_array(image)
    input_arr=np.array([input_arr])
    print("================FINISH LOADING TEST IMAGE==================")

    print("===================PREDICTING=====================")
    result=np.argmax(model.predict(input_arr),axis=1)
    print("YOUR RESULT AFTER GO THROUGH RESNET {}".format(result))
    print("==================END PREDICTING=====================")
