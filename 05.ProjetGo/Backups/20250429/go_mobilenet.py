# go_mobilenet.py - Version corrigée avec buildV2

from tensorflow import keras
from keras.models import Model
from keras import regularizers
from keras.layers import (
    Input, Dense, Conv2D, GlobalAveragePooling2D, Dropout, Flatten,
    Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D, Multiply,
    AveragePooling2D, Concatenate
)
from go_basenet import GoBaseNet


class GoMobileNet(GoBaseNet):

    def __init__(self, shape, n_class):
        super(GoMobileNet, self).__init__(shape, n_class)

    def build(self, block_num=2, filters=32, factor=4, se=True,
              drop_out_rate=0.5, activation='swish'):

        inputs = Input(shape=self.shape)
        x = self._conv_block(inputs, filters, (1, 1), activation='relu')

        for i in range(block_num):
            x = self._bottleneck_block(x, filters, (3, 3), factor, se, activation=activation)

        policy_head = self._conv_block(x, filters=1, kernel=(1, 1))
        policy_head = Flatten()(policy_head)
        policy_head = Activation('softmax', name='policy')(policy_head)

        value_head = GlobalAveragePooling2D()(x)
        value_head = Dense(50, kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer='he_normal')(value_head)

        value_head = Activation('relu')(value_head)
        value_head = Dropout(drop_out_rate)(value_head)
        value_head = Dense(1, activation='sigmoid', name='value',
                           kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer='he_normal')(value_head)

        model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
        return model
