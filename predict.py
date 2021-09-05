import cv2
import numpy as np
from google.colab.patches import cv2_imshow
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
    print('Predict using ResNet model for test file path {0}'.format(args.test_image)) # FIXME
    print('===========================')

    print("===================LOADING MODEL==========================")
    model=tf.keras.models.load_model('/content/best_model.h5')
    model.evaluate(val_generator)
    print("================FINISH LOADING MODE=======================")
    def pre_processing(img):
      img=tf.image.resize(img,[224,224])
      img/=225
      img=tf.expand_dims(img, axis=0)
      return img
    def predict(model, img_path, label_dict=label_dict):
      img=cv2.imread(img_path)
      cv2_imshow(img)
      img=pre_processing(img)
      prediction=label_dict[np.argmax(model.predict(img))]
      return prediction

    predict(model,'/content/image.png')
    print("====================END PREDICTING========================")
