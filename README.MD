# ResNet

<p align="center">
    <img src='https://i.ibb.co/XZmYNjK/Resnet-Logo.jpg' width=200 class="center">
</p>

Slide to tell how did we analyze and implement Resnet: [Click here](https://docs.google.com/presentation/d/1a18IyR1nc6GLTU5hibw3VabLVFbPjeE1/edit?usp=sharing&ouid=110166614717096573075&rtpof=true&sd=true)


Authors:
- Github: dark-kazansky, bdghuy, hoangcaobao, hoangduc199891, sonnymetvn
- Email: dark.kazansky@gmail.com, caobaohoang03@gmail.com, hoangduc199892@gmail.com, bdghuy@gmail.com, sonny.metvn@gmail.com

Advisors:
- Github: [bangoc123](https://github.com/bangoc123)
- Email: protonxai@gmail.com 

## I.  Set up environment
- Step 1: Make sure you have install conda like miniConda or Anaconda. 



- Step 2: from your terminal ```cd``` into Resnet folder then```python conda env create -f environment.yml```


## II.  Set up your dataset

- Run ```data.py``` to download cat&dog data set or you can prepair you dataset by yourself following instruction of folderStructure.py

## III. Training Process
Training script:


```python

python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```
Example:

```python

!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --patch-size 5 --image-size 150 --lr 0.0001 --epochs 200 --num-heads 12 

``` 
There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- ...

Notebook training: [<img src='https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667' width=100 class="left">](https://colab.research.google.com/drive/1cb7Nkkn5o_U7cVvnxXmfx-jdyoc4slhu?usp=sharing)

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

Your implementation
```
Epoch 00189: val_accuracy did not improve from 0.87700
Epoch 190/200
32/32 [==============================] - 55s 2s/step - loss: 0.0667 - accuracy: 0.9770 - val_loss: 0.8112 - val_accuracy: 0.8360

Epoch 00190: val_accuracy did not improve from 0.87700
Epoch 191/200
32/32 [==============================] - 55s 2s/step - loss: 0.0517 - accuracy: 0.9800 - val_loss: 0.7989 - val_accuracy: 0.8460

Epoch 00191: val_accuracy did not improve from 0.87700
Epoch 192/200
32/32 [==============================] - 54s 2s/step - loss: 0.0486 - accuracy: 0.9845 - val_loss: 0.6213 - val_accuracy: 0.8630

Epoch 00192: val_accuracy did not improve from 0.87700
Epoch 193/200
32/32 [==============================] - 55s 2s/step - loss: 0.0464 - accuracy: 0.9835 - val_loss: 0.5506 - val_accuracy: 0.8450

Epoch 00193: val_accuracy did not improve from 0.87700
Epoch 194/200
32/32 [==============================] - 55s 2s/step - loss: 0.0471 - accuracy: 0.9835 - val_loss: 1.0926 - val_accuracy: 0.8220

Epoch 00194: val_accuracy did not improve from 0.87700
Epoch 195/200
32/32 [==============================] - 55s 2s/step - loss: 0.0713 - accuracy: 0.9770 - val_loss: 1.0000 - val_accuracy: 0.8200

Epoch 00195: val_accuracy did not improve from 0.87700
Epoch 196/200
32/32 [==============================] - 55s 2s/step - loss: 0.0512 - accuracy: 0.9835 - val_loss: 1.9371 - val_accuracy: 0.6830

Epoch 00196: val_accuracy did not improve from 0.87700
Epoch 197/200
32/32 [==============================] - 55s 2s/step - loss: 0.0575 - accuracy: 0.9805 - val_loss: 1.1376 - val_accuracy: 0.7760

Epoch 00197: val_accuracy did not improve from 0.87700
Epoch 198/200
32/32 [==============================] - 55s 2s/step - loss: 0.0484 - accuracy: 0.9825 - val_loss: 0.6597 - val_accuracy: 0.8590

Epoch 00198: val_accuracy did not improve from 0.87700
Epoch 199/200
32/32 [==============================] - 55s 2s/step - loss: 0.0712 - accuracy: 0.9720 - val_loss: 1.4779 - val_accuracy: 0.8010

Epoch 00199: val_accuracy did not improve from 0.87700
Epoch 200/200
32/32 [==============================] - 55s 2s/step - loss: 0.0484 - accuracy: 0.9825 - val_loss: 0.6597 - val_accuracy: 0.8590

Epoch 00200: val_accuracy did not improve from 0.87700

```

## VI. Running Test

The ```best_model.h5``` of resnet50 is too large to commit on github so you can download it [here](https://drive.google.com/file/d/1pDfrAt7wHvDZrX4uolFxwE7KeUVTJvY8/view?usp=sharing). Then copy to base folder to load model.
In the ```./ResNet``` folder please run: ```predict.py --test-image "image-path"```

Or you can try this: [<img src='https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667' width=100 class="left">](https://colab.research.google.com/drive/1ySFObB6ZPgxJyq8G9_dHpQypExXAgAI2?usp=sharing&fbclid=IwAR0rEWKeq3m4Qmis2DZ5HkhCgZSEYj95kTImWqE_HZejUgYZRj9UiYdnjDM)

This is some results from us when we test for some regular dog or cat pictures:

<p align="center">
    <img src='https://i.imgur.com/UaAgqlO.jpeg' class="center">
</p>
<p align="center">
    <img src='https://i.imgur.com/t3jZt9D.jpeg' class="center">
</p>



