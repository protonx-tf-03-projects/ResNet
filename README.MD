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


**FIXME**

Training script:


```python

python train.py --epochs ${epochs} --input-lang en --target-lang vi --input-path ${path_to_en_text_file} --target-path ${path_to_vi_text_file}

```
**FIXME**

Example:

```python

!python train.py --train-folder ${train_folder} --valid-folder ${valid_folder} --num-classes 2 --patch-size 5 --image-size 150 --lr 0.0001 --epochs 200 --num-heads 12 

``` 
**FIXME**

There are some important arguments for the script you should consider when running it:

- `train-folder`: The folder of training data
- `valid-folder`: The folder of validation data
- ...

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

Your implementation
```
Epoch 117/120
8/8 [==============================] - 55s 2ms/step - loss: 0.0728 - acc: 0.9750 - val_loss: 1.9918 - val_acc: 0.7240
Epoch 118/120
8/8 [==============================] - 55s 2ms/step - loss: 0.0885 - acc: 0.9670 - val_loss: 0.4392 - val_acc: 0.8430
Epoch 119/120
8/8 [==============================] - 55s 2ms/step - loss: 0.0829 - acc: 0.9660 - val_loss: 0.4879 - val_acc: 0.8160
Epoch 120/120
8/8 [==============================] - 55s 2ms/step - loss: 0.0749 - acc: 0.9680 - val_loss: 0.5288 - val_acc: 0.8200

```

## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


