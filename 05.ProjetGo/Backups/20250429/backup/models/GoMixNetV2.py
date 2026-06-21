from tensorflow import keras
from keras import regularizers
from keras.models import Model
from keras.layers import (
    Input, Dense, Conv2D, GlobalAveragePooling2D, Dropout, Flatten,
    Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D,
    Concatenate, Multiply
)
from keras.activations import swish

def _se_block(input_tensor, filters, ratio=16, activation='relu', l2_lambda=0.0001):
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation=activation, use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2_lambda))(se)
    se = Dense(filters, activation='sigmoid', use_bias=False,
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2_lambda))(se)
    scaled = Multiply()([input_tensor, se])
    return scaled

def _conv_block(inputs, filters, kernel, activation='relu', l2_lambda=0.0001):
    x = Conv2D(filters, kernel, padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=regularizers.l2(l2_lambda),
               use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def _mixconv_block(inputs, filters, l2_lambda=0.0001):
    # Apply two depthwise convolutions with different kernel sizes
    x1 = DepthwiseConv2D((3, 3), padding='same',
                         depthwise_initializer='he_normal',
                         depthwise_regularizer=regularizers.l2(l2_lambda),
                         use_bias=False)(inputs)
    x2 = DepthwiseConv2D((5, 5), padding='same',
                         depthwise_initializer='he_normal',
                         depthwise_regularizer=regularizers.l2(l2_lambda),
                         use_bias=False)(inputs)
    x = Concatenate()([x1, x2])
    x = BatchNormalization()(x)
    x = Activation(swish)(x)
    return x

def _bottleneck_block(inputs, filters, factor, se=True, activation='relu', l2_lambda=0.0001):
    expanded_filters = filters * factor

    x = _conv_block(inputs, filters=expanded_filters, kernel=(1, 1), activation=activation, l2_lambda=l2_lambda)
    x = _mixconv_block(x, expanded_filters, l2_lambda)
    x = _conv_block(x, filters=filters, kernel=(1, 1), activation=activation, l2_lambda=l2_lambda)

    if se:
        x = _se_block(x, filters, 16, activation, l2_lambda)

    x = Add()([x, inputs])
    x = Activation(activation)(x)
    return x

def GoMixNetV2(input_shape, filters, factor, block_num, se=True, activation='relu', drop_out_rate=0.3, l2_lambda=0.0001):
    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, filters, (1, 1), activation=activation, l2_lambda=l2_lambda)

    for _ in range(block_num):
        x = _bottleneck_block(x, filters, factor, se, activation=activation, l2_lambda=l2_lambda)

    policy_head = _conv_block(x, filters=1, kernel=(1, 1), activation=activation, l2_lambda=l2_lambda)
    policy_head = Flatten()(policy_head)
    policy_head = Activation('softmax', name='policy')(policy_head)

    value_head = GlobalAveragePooling2D()(x)
    value_head = Dense(50, activation=activation,
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_lambda))(value_head)
    value_head = Dropout(drop_out_rate)(value_head)
    value_head = Dense(1, activation='sigmoid', name='value',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_lambda))(value_head)

    model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
    return model

if __name__ == "__main__":
    model = GoMixNetV2((19, 19, 31), filters=64, factor=6, block_num=2, se=True, drop_out_rate=0.5,activation='swish', l2_lambda=0.0001)
    model.summary()