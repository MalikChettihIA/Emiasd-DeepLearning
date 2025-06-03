# v2.0.2.1 Set SGD clipnorm=1.0

import time
import pandas as pd
import gc
import golois
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
from tensorflow.keras import optimizers

class LrLogger(Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_train_batch_end(self, batch, logs=None):
        # Récupère le learning rate de l'optimizer
        lr_schedule = self.model.optimizer.learning_rate

        if hasattr(lr_schedule, '__call__'):
            # Si c'est un scheduler (comme CosineDecay), évalue-le dynamiquement
            lr = float(K.get_value(lr_schedule(self.model.optimizer.iterations)))
        else:
            # Sinon (simple Variable/float), récupère la valeur
            lr = float(K.get_value(lr_schedule))

        self.lrs.append(lr)

def get_cosine_annealing_optimizer(initial_lr=0.002, decay_steps=10000, alpha=0.001):
    cosine_lr = CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        alpha=alpha
    )
    optimizer = optimizers.legacy.SGD(learning_rate=cosine_lr, momentum=0.9, nesterov=True, clipnorm=1.0)
    
    return optimizer

def get_cosine_annealing_restarts_optimizer(
    initial_lr=0.002,             # ↘ Démarrage plus stable
    first_decay_steps=2500,       # ↗ Un cycle ≈ 8 epochs
    t_mul=4.0,
    m_mul=0.5,                    # ↘ Amplitude diminue plus doucement
    alpha=1e-5                    # ↗ LR minimal utile
):
    cosine_lr = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=t_mul,
        m_mul=m_mul,
        alpha=alpha
    )

    optimizer = optimizers.legacy.SGD(learning_rate=cosine_lr, momentum=0.9, nesterov=True, clipnorm=1.0)
    return optimizer

def get_fixed_lr_optimizer(initial_lr=0.002, momentum=0.9):
    return optimizers.legacy.SGD(
        learning_rate=initial_lr,
        momentum=momentum,
        nesterov=True,
        clipnorm=1.0
    )

def lr_schedule(epoch):
    if epoch < 25:
        return 2e-2  # 0.02
    elif epoch < 50:
        return 2e-3  # 0.002
    elif epoch < 75:
        return 2e-3  # 0.0002
    else:
        return 2e-4  # 0.00002

def train_model(model, batch=32, initial_lr=0.002, momentum=0.9, policy_weight=1.0, value_weight=1.0, epochs=100, N=10000):
    start_time = time.time()
    
    # Configuration
    planes = 31
    moves = 361

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

    # Variable globale pour suivre la meilleure perte
    best_val_loss = float('inf')

    logger = LrLogger()

    #batches_per_epoch = N // batch  # ≈ 312
    #first_decay_steps = batches_per_epoch * 50  # ≈ 15000
    #optimizer = get_cosine_annealing_restarts_optimizer(
    #    initial_lr=0.005,
    #    first_decay_steps=first_decay_steps
    #)

    optimizer = get_fixed_lr_optimizer(initial_lr=initial_lr, momentum=momentum)
    #reduce_lr = ReduceLROnPlateau(
    #    monitor='policy_loss',
    #    factor=0.5,
    #    patience=10,
    #    min_lr=1e-5,
    #    verbose=1
    #)
    #reduce_lr.model = model

    model.compile(
        optimizer=optimizer,
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': policy_weight, 'value': value_weight},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )

    all_history = []
    val_loss_history = []

    best_model = None
    best_model_filename = None
    best_model_print = None

    for i in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Récupération des données
        golois.getBatch(input_data, policy, value, end, groups, i * N)

        history = model.fit(
            input_data,
            {'policy': policy, 'value': value},
            epochs=1,
            batch_size=batch,
            verbose=1,
            callbacks=[logger]
        )

        metrics = {key: val[0] for key, val in history.history.items()}
        metrics['epoch'] = i
        all_history.append(metrics)

        # Mise à jour manuelle du learning rate
        #new_lr = lr_schedule(i)
        #K.set_value(model.optimizer.lr, new_lr)

        # simulate end of epoch
        #logs = {'policy_loss': metrics['policy_loss']}
        #reduce_lr.on_epoch_end(i, logs)

        # vérifier le learning rate courant
        current_lr = float(keras.backend.get_value(model.optimizer.lr))
        print(f"Epoch {i} - LR: {current_lr}")

        if i % 5 == 0:
            gc.collect()

        if i % 20 == 0:
            # Évaluation du modèle sur les données de validation
            golois.getValidation(input_data, policy, value, end)
            val = model.evaluate(input_data, [policy, value], verbose=0, batch_size=batch)
            val_loss_history.append({
                'epoch': i,
                'val_policy_loss': val[1],
                'val_value_loss': val[2]
            })
            print(f"Validation: loss={val[0]}, policy_loss={val[1]:.4f}, value_loss={val[2]:.4f}")

            current_val_loss = val[0]  # loss globale
            if current_val_loss < best_val_loss :

                # Format propre du nom de fichier
                best_model_filename = f"best_model_epoch{i}_val{current_val_loss:.4f}.h5"
                best_model = model
                best_model_print = f"Best Model at epoch {i} : loss={val[0]}, policy_loss={val[1]:.4f}, value_loss={val[2]:.4f}"
                best_val_loss = current_val_loss

        # Affichage des métriques
        print(
            f"Epoch {i}/{epochs}: time={time.time() - epoch_start_time:.2f}s, "
            f"loss={metrics['loss']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"policy_categorical_accuracy={metrics['policy_categorical_accuracy']:.4f}, "
            f"value_mse={metrics['value_mse']:.4f}"
        )
    print(best_model_print)
    best_model.save(best_model_filename)

    total_time = time.time() - start_time
    return val, pd.DataFrame(all_history), pd.DataFrame(val_loss_history), total_time, logger.lrs