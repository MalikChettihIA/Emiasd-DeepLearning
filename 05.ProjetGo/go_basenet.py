from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Concatenate
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras import backend as K
from keras import regularizers


class GoBaseNet:

    def __init__(self, shape, n_class):
        self.shape = shape
        self.n_class = n_class

    def _conv_block(self, inputs, filters, kernel, activation=None, regul_l2=0.0001):
        x = Conv2D(filters, kernel, padding='same', kernel_regularizer=regularizers.l2(regul_l2),
                   use_bias=False, kernel_initializer='he_normal')(inputs)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x

    def _depthwise_conv_block(self, inputs, kernel, activation, regul_l2=0.0001):
        x = DepthwiseConv2D(kernel, depth_multiplier=1, padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(regul_l2),
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

    def _nested_bottleneck_block(self, inputs, filters, kernel, factor, squeeze, activation):
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

    def _nested_bottleneck_block_v2(self, inputs, activation='swish'):
        """
        Implémentation du Nested Bottleneck Residual Block de KataGo (Image 5)
        https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md#fixed-variance-initialization-and-one-batch-nor

        Architecture:
        Input (C channels)
        ↓
        Norm+Act+1x1 Conv → C/2 channels
        ↓
        [Premier sous-bloc résiduel]
        │ Norm+Act+3x3 Conv → C/2 channels
        │ ↓
        │ Norm+Act+3x3 Conv → C/2 channels
        │ ↓
        │ Add (skip connection C/2 channels)
        ↓
        [Deuxième sous-bloc résiduel]
        │ Norm+Act+3x3 Conv → C/2 channels
        │ ↓
        │ Norm+Act+3x3 Conv → C/2 channels
        │ ↓
        │ Add (skip connection C/2 channels)
        ↓
        Norm+Act+1x1 Conv → C channels
        ↓
        Add (main skip connection C channels)
        ↓
        Output
        """
        input_filters = inputs.shape[-1]
        reduced_filters = input_filters // 2

        # Skip connection principale (C channels)
        skip_connection_main = inputs

        # 1. Première réduction : C → C/2 channels
        x = self._conv_block(inputs, reduced_filters, (1, 1), activation)

        # 2. Premier sous-bloc résiduel (C/2 channels)
        skip_connection_sub1 = x

        # Première paire de convolutions 3x3
        x = self._conv_block(x, reduced_filters, (3, 3), activation)
        x = self._conv_block(x, reduced_filters, (3, 3), activation)

        # Addition de la première connexion résiduelle du sous-bloc
        x = Add()([x, skip_connection_sub1])

        # 3. Deuxième sous-bloc résiduel (C/2 channels)
        skip_connection_sub2 = x

        # Deuxième paire de convolutions 3x3
        x = self._conv_block(x, reduced_filters, (3, 3), activation)
        x = self._conv_block(x, reduced_filters, (3, 3), activation)

        # Addition de la deuxième connexion résiduelle du sous-bloc
        x = Add()([x, skip_connection_sub2])

        # 4. Expansion finale : C/2 → C channels
        x = self._conv_block(x, input_filters, (1, 1), activation)

        # 5. Addition de la connexion résiduelle principale (C + C → C)
        x = Add()([x, skip_connection_main])

        return x


    def get_iteration_block(self, inputs, filters, kernel, activation):
        x = self._conv_block(inputs, filters, kernel, activation=activation)
        x = self._conv_block(x, filters, kernel, activation=activation)
        x = Add()([inputs, x])
        return x


    # --- MixDepthwise Block ---
    def _mix_depthwise_conv(self, inputs, activation, regul_l2=0.0001):
        # Le nombre de canaux est déterminé par l'input
        input_channels = inputs.shape[-1]
        channels_3x3 = input_channels // 2
        channels_5x5 = input_channels - channels_3x3

        x1 = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=1, use_bias=False,
                             depthwise_regularizer=regularizers.l2(regul_l2))(inputs)
        x2 = DepthwiseConv2D((5, 5), padding='same', depth_multiplier=1, use_bias=False,
                             depthwise_regularizer=regularizers.l2(regul_l2))(inputs)

        # Sélection des canaux appropriés
        x = Concatenate()([x1[:, :, :, :channels_3x3], x2[:, :, :, :channels_5x5]])
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        return x

    def _bottleneck_mix_block(self, inputs, filters, factor, squeeze, activation):
        input_filters = inputs.shape[-1]
        expansion = int(input_filters * factor)  # ✅ Basé sur input_filters

        # Expansion 1x1
        x = self._conv_block(inputs, expansion, (1, 1), activation)

        # MixDepthwise convolution (ne prend que inputs et activation)
        x = self._mix_depthwise_conv(x, activation)  # ✅ Plus de paramètre filters

        # Squeeze & Excitation optionnel
        if squeeze:
            x = self._squeeze_block(x)

        # Projection 1x1 vers les canaux de sortie
        x = self._conv_block(x, filters, (1, 1), activation)

        # Connexion résiduelle seulement si dimensions compatibles
        if input_filters == filters:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass
