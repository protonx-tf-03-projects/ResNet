# ResNet


<p align="center">
    <img src='https://miro.medium.com/max/868/1*ZSvSAACQVvUHU1qS-mkYKw.png' width=200 class="center">
</p>

Description about your project. Why do you choose to build this?  <--- **FIXME**

Slide about your project (if it's available) <--- **FIXME**

<p align="center">
    <img src='https://developer.ridgerun.com/wiki/images/f/f5/Resnet_architecture.png' width=677 class="center">
</p>

Authors:
- Github: dark-kazansky, bdghuy, hoangcaobao, hoangduc199891, sonnymetvn
- Email: dark.kazansky@gmail.com, caobaohoang03@gmail.com, hoangduc199892@gmail.com, bdghuy@gmail.com, sonny.metvn@gmail.com

Advisors:
- Github: advisor github name <--- **FIXME**
- Email: advisor emails <--- **FIXME**

## I.  Set up environment
- Step 1: in your terminal, type: git clone https://github.com/protonx-tf-03-projects/ResNet.git

- Step 2: cd into ResNet, and type this command line into your terminal.
```
conda env create -f environment.yml
```

- Step 3: run conda environment using this command:

```
conda activate resnet_team
``` 

## II.  Set up your dataset

- Guide user how to download your data and set the data pipeline 
- II.1. create a folder named ```data```
- II.2. in folder "data", create 2 folders train and validation in the data folder (which was created already). Then Please copy your images with the corresponding names into these folders.

- ```train``` folder was used for the training process
- ```validation``` folder was used for validating training result after each epoch

- Structure of these folders.
```
train/
...cats/
......cat.0.jpg
......cat.1.jpg
...dogs/
......dog.0.jpg
......dog.1.jpg
```
```
validation/
......cat.0.jpg
......cat.1.jpg
...dogs/
......dog.0.jpg
......dog.1.jpg
```
- References: [NLP](https://github.com/bangoc123/transformer) and [CV](https://github.com/bangoc123/mlp-mixer)

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

**FIXME**

Your implementation
```
Epoch 7/10
782/782 [==============================] - 261s 334ms/step - loss: 0.8315 - acc: 0.8565 - val_loss: 0.8357 - val_acc: 0.7978
Epoch 8/10
782/782 [==============================] - 261s 334ms/step - loss: 0.3182 - acc: 0.8930 - val_loss: 0.6161 - val_acc: 0.8047
Epoch 9/10
782/782 [==============================] - 261s 333ms/step - loss: 1.1965 - acc: 0.8946 - val_loss: 3.9842 - val_acc: 0.7855
Epoch 10/10
782/782 [==============================] - 261s 333ms/step - loss: 0.4717 - acc: 0.8878 - val_loss: 0.4894 - val_acc: 0.8262

```

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


