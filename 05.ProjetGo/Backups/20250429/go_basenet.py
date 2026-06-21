from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Concatenate
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras import backend as K
from keras import regularizers


class GoBaseNet:

    def __init__(self, shape, n_class):
        self.shape = shape
        self.n_class = n_class

    def _conv_block(self, inputs, filters, kernel, activation=None):
        x = Conv2D(filters, kernel, padding='same', kernel_regularizer=regularizers.l2(0.0001),
                   use_bias=False, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x

    def _depthwise_conv_block(self, inputs, kernel, activation):
        x = DepthwiseConv2D(kernel, depth_multiplier=1, padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001),
                            kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def _squeeze_block(self, input, ratio=16):
        filters = input.shape[-1]
        se = GlobalAveragePooling2D()(input)
        se = Reshape((1, 1, filters))(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)
        x = Multiply()([input, se])
        return x

    def _bottleneck_block(self, inputs, filters, kernel, factor, squeeze, activation):
        # Expansion basée sur les canaux d'entrée
        input_filters = inputs.shape[-1]
        expansion = int(input_filters * factor)

        # Expansion 1x1
        x = self._conv_block(inputs, expansion, (1, 1), activation)

        # Depthwise convolution
        x = self._depthwise_conv_block(x, kernel, activation)

        # Squeeze & Excitation optionnel
        if squeeze:
            x = self._squeeze_block(x)

        # Projection 1x1 vers les canaux de sortie
        x = self._conv_block(x, filters, (1, 1), activation)

        # ✅ Connexion résiduelle seulement si dimensions compatibles
        if input_filters == filters:
            x = Add()([x, inputs])

        return x
    #def _bottleneck_block(self, inputs, filters, kernel, factor, squeeze, activation):
    #    expension = int(filters * factor)
    #    x = self._conv_block(inputs, expension, (1, 1), activation)
    #    x = self._depthwise_conv_block(x, kernel, activation)
    #    if squeeze:
    #        x = self._squeeze_block(x)
    #    x = self._conv_block(x, filters, (1, 1), activation)
    #    x = Add()([x, inputs])
    #    return x

    def get_iteration_block(self, inputs, filters, kernel, activation):
        x = self._conv_block(inputs, filters, kernel, activation=activation)
        x = self._conv_block(x, filters, kernel, activation=activation)
        x = Add()([inputs, x])
        return x


    # --- MixDepthwise Block ---
    def _mix_depthwise_conv(self, inputs, filters, activation):

        # Half filters with 3x3, half with 5x5 kernels
        filters_3x3 = filters // 2
        filters_5x5 = filters - filters_3x3

        x1 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, use_bias=False,
                             depthwise_regularizer=regularizers.l2(0.0001))(inputs)
        x2 = DepthwiseConv2D((5, 5), padding='same', depth_multiplier=1, use_bias=False,
                             depthwise_regularizer=regularizers.l2(0.0001))(inputs)
        x = Concatenate()([x1[:, :, :, :filters_3x3], x2[:, :, :, :filters_5x5]])
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def _bottleneck_mix_block(self, inputs, filters, factor, squeeze, activation):
        expension = int(filters * factor)
        x = self._conv_block(inputs, expension, (1, 1), activation)
        x = self._mix_depthwise_conv(x, filters, activation)
        if squeeze:
            x = self._squeeze_block(x)
        x = self._conv_block(x, filters, (1, 1), activation)
        x = Add()([x, inputs])
        return x

    def build(self):
        pass
