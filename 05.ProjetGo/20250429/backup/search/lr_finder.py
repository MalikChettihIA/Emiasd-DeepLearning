import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
import golois
from go_mobilenet import GoMobileNet
from keras_lr_finder import LRFinder


if __name__ == "__main__":

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()

    # Configuration
    planes = 31
    moves = 361
    N = 100000

    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype ('float32')

    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical (policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype ('float32')

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype ('float32')

    groups = np.zeros((N, 19, 19, 1))
    groups = groups.astype ('float32')

    # Get Validation Data
    print ("getValidation", flush = True)
    golois.getValidation (input_data, policy, value, end)

    optimizer = optimizers.legacy.Adam(
        learning_rate=0.002,
    )

    model.compile(
        optimizer=optimizer,
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )

    lr_finder = LRFinder(model)

    lr_finder.find(
        input_data,
        {'policy': policy, 'value': value},
        start_lr=0.00001, end_lr=1, batch_size=32, epochs=1)
    lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
    print(f'Best Learning rate is {lr_finder.get_best_lr(sma=1, n_skip_beginning=20, n_skip_end=5)}')