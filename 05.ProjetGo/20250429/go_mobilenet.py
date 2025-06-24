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

    def buildV2(self, block_num=2, filters=32, factor=4, se=True,
                drop_out_rate=0.5, activation='swish'):
        """
        Version V2 avec features spécialisées pour améliorer la value head
        """

        # ✅ Utilisation correcte d'Input (pas input)
        inputs = Input(shape=self.shape)

        # Trunk partagé
        x = self._conv_block(inputs, filters, (1, 1), activation='relu')

        for i in range(block_num):
            x = self._bottleneck_block(x, filters, (3, 3), factor, se, activation=activation)

        # ============ FEATURES SPÉCIALISÉES ============

        # Features spécialisées pour POLICY (spatiales, locales)
        policy_features = self._conv_block(x, filters // 2, (1, 1), activation=activation)
        policy_features = self._conv_block(policy_features, filters // 4, (3, 3), activation=activation)

        # Features spécialisées pour VALUE (globales, évaluatives)
        value_features = self._conv_block(x, filters // 4, (1, 1), activation=activation)
        # Capture patterns plus larges pour évaluation globale
        value_features = DepthwiseConv2D((5, 5), padding='same',
                                         depth_multiplier=1, use_bias=False,
                                         kernel_regularizer=regularizers.l2(0.0001))(value_features)
        value_features = BatchNormalization()(value_features)
        value_features = Activation(activation)(value_features)

        # ============ HEADS SPÉCIALISÉES ============

        # Policy Head (utilise policy_features)
        policy_head = self._conv_block(policy_features, filters=1, kernel=(1, 1))
        policy_head = Flatten()(policy_head)
        policy_head = Activation('softmax', name='policy')(policy_head)

        # Value Head améliorée (utilise value_features)
        value_head = GlobalAveragePooling2D()(value_features)

        # Architecture value head plus puissante
        value_head = Dense(128, kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer='he_normal')(value_head)
        value_head = Activation(activation)(value_head)
        value_head = Dropout(drop_out_rate)(value_head)

        value_head = Dense(64, kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer='he_normal')(value_head)
        value_head = Activation(activation)(value_head)

        # Sortie avec tanh au lieu de sigmoid pour [-1, 1]
        value_head = Dense(1, activation='tanh', name='value',
                           kernel_regularizer=regularizers.l2(0.0001),
                           kernel_initializer='he_normal')(value_head)

        # ✅ Création du modèle avec inputs (pas input)
        model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
        return model

    def buildV3(self, block_num=2, filters=32, factor=4, se=True,
                               drop_out_rate=0.5, activation='swish'):
        """
        Version V3 avec mécanisme d'attention pour features spécialisées
        """

        inputs = Input(shape=self.shape)

        # Trunk partagé
        x = self._conv_block(inputs, filters, (1, 1), activation='relu')

        for i in range(block_num):
            x = self._bottleneck_block(x, filters, (3, 3), factor, se, activation=activation)

        # ============ ATTENTION MECHANISM ============

        # Attention pour Policy (focus sur patterns locaux)
        policy_attention = Conv2D(1, 1, activation='sigmoid',
                                  padding='same', name='policy_attention')(x)
        policy_features = Multiply()([x, policy_attention])
        policy_features = self._conv_block(policy_features, filters // 2, (1, 1), activation=activation)

        # Attention pour Value (focus sur évaluation globale)
        value_attention = Conv2D(1, 1, activation='sigmoid',
                                 padding='same', name='value_attention')(x)
        value_features = Multiply()([x, value_attention])
        value_features = self._conv_block(value_features, filters // 4, (1, 1), activation=activation)

        # ============ HEADS AVEC ATTENTION ============

        # Policy Head
        policy_head = self._conv_block(policy_features, filters=1, kernel=(1, 1))
        policy_head = Flatten()(policy_head)
        policy_head = Activation('softmax', name='policy')(policy_head)

        # Value Head
        value_head = GlobalAveragePooling2D()(value_features)
        value_head = Dense(128, activation=activation,
                           kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = Dropout(drop_out_rate)(value_head)
        value_head = Dense(1, activation='tanh', name='value',
                           kernel_regularizer=regularizers.l2(0.0001))(value_head)

        model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
        return model