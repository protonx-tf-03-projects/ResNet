import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input,MaxPooling2D,GlobalAveragePooling2D, Dense
from tensorflow.keras import Model

def bn_rl(tensor):
  norm=BatchNormalization(axis=3)(tensor)

  return ReLU()(norm)

def conv_bn_rl(tensor,filters,kernel_size=3,stride=1,padding='same',use_activation=True):
  conv=Conv2D(filters,kernel_size,stride,padding)(tensor)
  norm=BatchNormalization(axis=3)(conv)
  return ReLU()(norm) if use_activation else norm

def bn_relu_conv(tensor,filters, kernel_size=3, stride=1,padding='same',use_bias=False):
  pre_activation=bn_rl(tensor)
  return Conv2D(filters,kernel_size,stride,padding,use_bias=use_bias)(pre_activation)

def add_shortcut(residual, tensor,bn=False):
  tensor_channel=tensor.shape[3]
  residual_channel=residual.shape[3]
  shortcut=tensor
  if tensor_channel != residual_channel:
    stride=int(tensor.shape[1]/residual.shape[1])
    shortcut=Conv2D(residual_channel,1,stride)(tensor)
    if bn:
      shortcut=BatchNormalization()(shortcut)
    
  return Add()([residual, shortcut])

# Basic block
def basic_block(tensor, filters, stride=1,is_first_block=False):
  conv1=conv_bn_rl(tensor, filters,stride=stride)
  residual=conv_bn_rl(conv1, filters,use_activation=False)

  return ReLU()(add_shortcut(residual,tensor,bn=True))

def basic_block_V2(tensor, filters,stride=1,is_first_block=False):
  if is_first_block:
    # activated input
    conv=Conv2D(filters,3,1,padding='same',use_bias=False)(tensor)
  else:
    conv=bn_relu_conv(tensor,filters,stride=stride)
  residual=bn_relu_conv(conv, filters,use_bias=True)

  return add_shortcut(residual,tensor)

# Bottleneck block
def bottleneck_block(tensor, filters, stride=1,is_first_block=False):
  conv1=conv_bn_rl(tensor, filters,1,stride)
  conv2=conv_bn_rl(conv1, filters)
  conv3=conv_bn_rl(conv2, 4*filters,1,use_activation=False)

  return ReLU()(add_shortcut(conv3,tensor, bn=True))

def bottleneck_block_V2(tensor, filters,stride=1,is_first_block=False):
  if is_first_block:
    # not use activation
    conv=Conv2D(filters,1,1,padding='same',use_bias=False)(tensor)
  else:
    conv=bn_relu_conv(tensor, filters,1,stride=stride)

  conv=bn_relu_conv(conv, filters)
  residual=bn_relu_conv(conv, filters*4,1,use_bias=True)

  return add_shortcut(residual,tensor)


def stage(tensor, filters,block,num_blocks, down_sampling=True):
  ''' 
  Đầu vào:
    filters:
    block: basic_block hoặc bottleneck_block
    num_blocks: số lần lặp lại của một block trong mỗi stage

    tensor: kích thước [None,H, W, C]
  Đầu ra:
    tensor: kích thước [None,H', W', C']
  '''
  # block đầu tiên của mỗi stage
  x=block(tensor,filters, stride= (2 if down_sampling else 1),is_first_block=True)
    
  # block thứ 2 trở đi
  for i in range(num_blocks-1):
    x=block(x,filters)
  return x if block.__name__[-2:]!='V2' else bn_rl(x)


def build(input_shape,num_classes,block,arr):
  ''' 
  Đầu vào:
    input_shape: 
                 vd: (224,224,3)
    num_classes: số class cần phân loại 
                 vd: 1000
    block: basic_block hoặc bottleneck_block
    arr: mảng 4 phần tử [r2, r3, r4, r5], mỗi phần tử là số lần lặp lại của một block trong lớp tương ứng.
         vd: [2,2,2,2] cho resnet18

  Đầu ra:
    trả về model
  '''
  input=Input(input_shape)
  f=64
  # conv1
  x=conv_bn_rl(input,f,7,2)
  x = MaxPooling2D(pool_size=(3, 3), strides=2,padding='same')(x)
  # conv2_x, conv3_x, conv4_x, conv5_x
  for e,r in enumerate(arr):
    x=stage(x,f,block,r,e!=0)
    f*=2
  
  x=GlobalAveragePooling2D()(x)
  output=Dense(num_classes,activation='softmax')(x)
  model=Model(input,output)

  return model

# Resnet V1
def resnet18(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,basic_block,[2,2,2,2])

def resnet34(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,basic_block,[3,4,6,3])

def resnet50(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block,[3,4,6,3])

def resnet101(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block,[3,4,23,3])

def resnet152(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block,[3,8,36,3])

# Resnet V2
def resnet18V2(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,basic_block_V2,[2,2,2,2])

def resnet34V2(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,basic_block_V2,[3,4,6,3])

def resnet50V2(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block_V2,[3,4,6,3])

def resnet101V2(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block_V2,[3,4,23,3])

def resnet152V2(input_shape=(224,224,3),num_classes=1000):
  return build(input_shape,num_classes,bottleneck_block_V2,[3,8,36,3])