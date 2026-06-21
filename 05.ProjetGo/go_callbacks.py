from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

class LrLogger(Callback):
    """Callback pour logger le learning rate à chaque batch"""

    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_train_batch_end(self, batch, logs=None):
        lr_schedule = self.model.optimizer.learning_rate

        if hasattr(lr_schedule, '__call__'):
            lr = float(K.get_value(lr_schedule(self.model.optimizer.iterations)))
        else:
            lr = float(K.get_value(lr_schedule))

        self.lrs.append(lr)