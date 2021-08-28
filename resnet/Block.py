from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_uniform

class BasicBlock(Layer):
    def __init__(self, f, stride):
        super(BasicBlock, self).__init__()

        self.expansion = 1
        self.Conv2a = Conv2D(f, kernel_size=(3, 3), strides=stride, padding='same',
                             kernel_initializer=glorot_uniform(seed=0))
        self.Bn2a = BatchNormalization(axis=3)

        self.Conv2b = Conv2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=glorot_uniform(seed=0))
        self.Bn2b = BatchNormalization(axis=3)

    def call(self, input_tensor):
        X = self.Conv2a(input_tensor)
        X = self.Bn2a(X)
        X = Activation('relu')(X)

        X = self.Conv2b(X)
        X = self.Bn2b(X)

        return X

class BottleNeckBlock(Layer):
  def __init__(self, f, stride):
    super(BottleNeckBlock, self).__init__(name='')

    self.expansion = 4
    self.Conv2a = Conv2D(f, kernel_size= (1, 1), strides = stride, padding = 'valid', kernel_initializer = glorot_uniform(seed=0))
    self.Bn2a = BatchNormalization(axis = 3)

    self.Conv2b = Conv2D(f, kernel_size= (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))
    self.Bn2b = BatchNormalization(axis = 3)

    self.Conv2c = Conv2D(f*4, kernel_size= (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))
    self.Bn2c = BatchNormalization(axis = 3)

  def call(self, input_tensor):
    X = self.Conv2a(input_tensor)
    X = self.Bn2a(X)
    X = Activation('relu')(X)

    X = self.Conv2b(X)
    X = self.Bn2b(X)
    X = Activation('relu')(X)

    X = self.Conv2c(X)
    X = self.Bn2c(X)

    return X

class ConvolutionalBlock(Layer):
  def __init__(self, f, use_BottleNeck = False):
      super(ConvolutionalBlock, self).__init__(name='')

      if (use_BottleNeck):
        self.ConBlock = BottleNeckBlock(f, stride = 2)
        self.expansion = 4
      else:
        self.ConBlock = BasicBlock(f, stride = 2)
        self.expansion = 1
      self.downsample = Sequential()
      self.downsample.add(Conv2D(f*self.expansion, kernel_size= (1, 1), strides = 2, padding = 'valid'))
      self.downsample.add(BatchNormalization(axis = 3))
  def call(self, input_tensor):
    ##### MAIN PATH #####
    X = self.ConBlock(input_tensor)

    ##### SHORTCUT PATH #### (21 lines)
    X_shortcut = self.downsample(input_tensor)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (2 lines)
    X += X_shortcut
    X = Activation('relu')(X)

    return X


class BuildingBlock(Layer):
    def __init__(self, filter, stride, use_BottleNeck=False, use_DownSample=False):
        super(BuildingBlock, self).__init__()

        if (use_BottleNeck):
            self.Block = BottleNeckBlock(filter, stride)
        else:
            self.Block = BasicBlock(filter, stride)

        if use_DownSample:
            self.Down = ConvolutionalBlock(filter, use_BottleNeck=use_BottleNeck)
        else:
            self.Down = 0

    def call(self, input_tensor):
        if (self.Down != 0):
            X = self.Down(input_tensor)
        else:
            X = self.Block(input_tensor)
            X += input_tensor
            X = Activation('relu')(X)
        return X