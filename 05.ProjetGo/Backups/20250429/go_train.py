import time
import pandas as pd
import gc
import golois
import numpy as np

from tensorflow import keras
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
from training_monitor import TrainingMonitor


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


def apply_go_transformation(board, policy, transform_id):
    """
    Applique une transformation spécifique au plateau et à la politique
    Modifie les arrays sur place pour éviter la surcharge mémoire
    """
    if transform_id == 0:
        # Identité - pas de transformation
        return

    elif transform_id == 1:
        # Rotation 90° horaire
        board[:] = np.rot90(board, k=-1, axes=(0, 1))
        if len(policy.shape) == 1 and len(policy) == 361:  # Vecteur (361,)
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=-1, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:  # One-hot (batch, 361)
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=-1, axes=(0, 1))
            policy[:] = policy_2d.flatten()

    elif transform_id == 2:
        # Rotation 180°
        board[:] = np.rot90(board, k=2, axes=(0, 1))
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=2, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=2, axes=(0, 1))
            policy[:] = policy_2d.flatten()

    elif transform_id == 3:
        # Rotation 270° horaire
        board[:] = np.rot90(board, k=1, axes=(0, 1))
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=1, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=1, axes=(0, 1))
            policy[:] = policy_2d.flatten()

    elif transform_id == 4:
        # Miroir horizontal
        board[:] = np.flip(board, axis=1)
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=1)
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=1)
            policy[:] = policy_2d.flatten()

    elif transform_id == 5:
        # Miroir vertical
        board[:] = np.flip(board, axis=0)
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=0)
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=0)
            policy[:] = policy_2d.flatten()

    elif transform_id == 6:
        # Transpose (diagonal principal)
        board[:] = np.transpose(board, (1, 0, 2))
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.transpose(policy_2d, (1, 0))
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.transpose(policy_2d, (1, 0))
            policy[:] = policy_2d.flatten()

    elif transform_id == 7:
        # Anti-diagonal (transpose + double flip)
        board[:] = np.flip(np.flip(np.transpose(board, (1, 0, 2)), axis=0), axis=1)
        if len(policy.shape) == 1 and len(policy) == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(np.flip(np.transpose(policy_2d, (1, 0)), axis=0), axis=1)
            policy[:] = policy_2d.flatten()
        elif len(policy.shape) == 2 and policy.shape[-1] == 361:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(np.flip(np.transpose(policy_2d, (1, 0)), axis=0), axis=1)
            policy[:] = policy_2d.flatten()


def get_cosine_schedule_with_fixed_restarts(config):
    # Calcul des batches par époque
    batches_per_epoch = config['train']['N'] // config['train']['batch']  # 10000 // 32 ≈ 312

    # Ajuster le nombre de batches si augmentation
    if config['augmentation']['use_augmentation']:
        batches_per_epoch *= config['augmentation']['nb_rotation']

    total_batches = batches_per_epoch * config['train']['epochs']

    # Étapes par cycle (même longueur à chaque fois)
    first_decay_steps = total_batches // config['train']['nb_cosinedecay_cycle']

    cosine_lr = CosineDecayRestarts(
        initial_learning_rate=config['train']['initial_lr'],
        first_decay_steps=first_decay_steps,
        t_mul=config['train']['t_mul_lr'],
        m_mul=config['train']['m_mul_lr'],
        alpha=config['train']['alpha_lr']
    )

    if config['train']['initial_lr'] != config['train']['alpha_lr']:
        optimizer = optimizers.AdamW(learning_rate=cosine_lr, clipnorm=1.0)
    else:
        optimizer = optimizers.legacy.Adam(learning_rate=config['train']['initial_lr'], clipnorm=1.0)

    return optimizer


def train_model(model, config):
    """
    Fonction d'entraînement simplifiée avec monitoring unifié
    """
    monitor = TrainingMonitor(
        project_name="Go-Project",
        entity='Emiasd',
        experiment_name=f"{config.get('experiment_name', 'experiment')}"
    )
    
    # Initialiser WandB avec la configuration
    monitor.initialize_wandb(
        config=config,
        tags=["training", "mobilenet", "go"],
        notes=f"Go training"
    )
    
    # Log des informations du modèle
    monitor.log_model_info(model)

    # ============================================================================
    # CONFIGURATION DE L'ENTRAÎNEMENT
    # ============================================================================
    
    start_time = time.time()
    
    # Info augmentation
    if config['augmentation']['use_augmentation']:
        print(f"🔄 Augmentation de données activée: {config['augmentation']['nb_rotation']} transformations par époque")
    else:
        print("❌ Entraînement sans augmentation de données")

    # Configuration des données
    planes = 31
    moves = 361

    input_data = np.random.randint(2, size=(config['train']['N'], 19, 19, planes))
    input_data = input_data.astype('float32')

    policy = np.random.randint(moves, size=(config['train']['N'],))
    policy = keras.utils.to_categorical(policy)

    value = np.random.randint(2, size=(config['train']['N'],))
    value = value.astype('float32')

    end = np.random.randint(2, size=(config['train']['N'], 19, 19, 2))
    end = end.astype('float32')

    groups = np.zeros((config['train']['N'], 19, 19, 1))
    groups = groups.astype('float32')

    # Get Validation Data
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    # Configuration du modèle
    logger = LrLogger()
    optimizer = get_cosine_schedule_with_fixed_restarts(config)

    model.compile(
        optimizer=optimizer,
        loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
        loss_weights={'policy': config['train']['policy_weight'], 'value': config['train']['value_weight']},
        metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
    )

    # Variables pour l'historique
    all_history = []
    val_loss_history = []

    # ============================================================================
    # BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    # ============================================================================

    for i in range(1, config['train']['epochs'] + 1):
        epoch_start_time = time.time()
        
        # Récupération des données
        golois.getBatch(input_data, policy, value, end, groups, i * config['train']['N'])

        # Sauvegarder les données originales pour la boucle d'augmentation
        original_input = input_data.copy()
        original_policy = policy.copy()
        original_value = value.copy()

        # Variables pour accumuler les métriques de toutes les transformations
        epoch_histories = []
        transform_names = ["Original", "Rot90°", "Rot180°", "Rot270°",
                           "MirrorH", "MirrorV", "DiagMain", "DiagAnti"]

        # Boucle d'augmentation intégrée
        augmentation_range = range(config['augmentation']['nb_rotation']) if config['augmentation']['use_augmentation'] else range(1)

        if config['augmentation']['use_augmentation']:
            print(f"\n📊 Époque {i}/{config['train']['epochs']} - Entraînement sur {config['augmentation']['nb_rotation']} transformations:")

        for j in augmentation_range:
            transform_start_time = time.time()

            # Restaurer les données originales avant chaque transformation
            input_data[:] = original_input
            policy[:] = original_policy
            value[:] = original_value

            # Appliquer la transformation j à tout le batch (si j > 0)
            if j > 0:  # 0 = identité, pas besoin de transformer
                for sample_idx in range(config['train']['N']):
                    apply_go_transformation(input_data[sample_idx], policy[sample_idx], j)

            # Entraînement sur cette transformation
            history = model.fit(
                input_data,
                {'policy': policy, 'value': value},
                epochs=1,
                batch_size=config['train']['batch'],
                verbose=0 if config['augmentation']['use_augmentation'] and j > 0 else 1,
                callbacks=[logger]
            )

            # Enregistrer les métriques de cette transformation
            transform_metrics = {key: val[0] for key, val in history.history.items()}
            epoch_histories.append(transform_metrics)

            # Affichage pour chaque transformation si augmentation activée
            if config['augmentation']['use_augmentation']:
                transform_time = time.time() - transform_start_time
                print(f"  🔀 {transform_names[j]:8s} (T{j}): "
                      f"loss={transform_metrics['loss']:.4f}, "
                      f"policy_loss={transform_metrics['policy_loss']:.4f}, "
                      f"value_loss={transform_metrics['value_loss']:.4f}, "
                      f"policy_categorical_accuracy={transform_metrics['policy_categorical_accuracy']:.4f}, "
                      f"value_mse={transform_metrics['value_mse']:.4f}, "
                      f"time={transform_time:.1f}s")

        # Calculer les métriques moyennes de l'époque
        if config['augmentation']['use_augmentation']:
            avg_metrics = {}
            for key in epoch_histories[0].keys():
                avg_metrics[key] = np.mean([h[key] for h in epoch_histories])
            metrics = avg_metrics
        else:
            metrics = {key: val[0] for key, val in history.history.items()}

        metrics['epoch'] = i
        all_history.append(metrics)

        # ============================================================================
        # MONITORING ET LOGGING UNIFIÉ
        # ============================================================================

        # Préparer les métriques de timing
        epoch_time = time.time() - epoch_start_time
        timing_metrics = {
            "epoch_duration": epoch_time,
            "total_elapsed": time.time() - start_time,
            "augmentation_enabled": config['augmentation']['use_augmentation']
        }

        # Log des métriques d'époque vers WandB
        current_lr = logger.lrs[-1] if logger.lrs else None
        monitor.log_epoch_metrics(
            epoch=i,
            train_metrics=metrics,
            learning_rate=current_lr,
            timing=timing_metrics
        )

        # Affichage des métriques avec info augmentation
        aug_info = f" ({config['augmentation']['nb_rotation']} transforms avg)" if config['augmentation']['use_augmentation'] else ""
        print(
            f"\n✅ Époque {i}/{config['train']['epochs']}{aug_info}: time={epoch_time:.2f}s, "
            f"loss={metrics['loss']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"policy_categorical_accuracy={metrics['policy_categorical_accuracy']:.4f}, "
            f"value_mse={metrics['value_mse']:.4f}"
        )

        # Nettoyage mémoire périodique
        if i % 5 == 0:
            gc.collect()

        # ============================================================================
        # VALIDATION ET MONITORING AVANCÉ
        # ============================================================================

        if i==1 or i % 10 == 0 or i == config['train']['epochs']+1 :
            # Restaurer les données originales pour la validation
            input_data[:] = original_input
            policy[:] = original_policy
            value[:] = original_value

            # Évaluation du modèle sur les données de validation (TOUJOURS sans augmentation)
            print("\n🔍 Évaluation sur données de validation (non augmentées)...")
            golois.getValidation(input_data, policy, value, end)
            val = model.evaluate(input_data, [policy, value], verbose=1, batch_size=config['train']['batch'])

            val_loss_history.append({
                'epoch': i,
                'val_total_loss': val[0],  # Total loss
                'val_policy_loss': val[1],  # Policy loss
                'val_value_loss': val[2],  # Value loss
                'val_policy_accuracy': val[3],  # Policy accuracy
                'val_value_mse': val[4]  # Value MSE
            })

            # Préparer les métriques de validation
            val_metrics = {
                'val_total_loss': val[0],
                'val_policy_loss': val[1],
                'val_value_loss': val[2],
                'val_policy_categorical_accuracy': val[3],
                'val_value_mse': val[4]
            }

            # Log vers WandB
            monitor.log_epoch_metrics(
                epoch=i,
                train_metrics=metrics,
                val_metrics=val_metrics,
                learning_rate=current_lr,
                timing=timing_metrics
            )

            # Affichage des métriques de validation
            print(
                f"📈 Validation Époque {i}/{config['train']['epochs']}: "
                f"loss={val[0]:.4f}, "
                f"policy_loss={val[1]:.4f}, "
                f"value_loss={val[2]:.4f}, "
                f"policy_categorical_accuracy={val[3]:.4f}, "
                f"value_mse={val[4]:.4f}"
            )

            # Sauvegarde conditionnelle du meilleur modèle
            monitor.log_model_checkpoint(
                model=model,
                epoch=i,
                metrics=val_metrics,
                save_condition=True  # Toujours vérifier si c'est le meilleur
            )

        # ============================================================================
        # MONITORING DES GRADIENTS
        # ============================================================================

        if i % 10 == 0:  # Check gradients moins fréquemment
            try:
                # Préparer un échantillon pour l'analyse des gradients
                sample_size = min(32, config['train']['batch'])
                sample_inputs = input_data[:sample_size]
                sample_targets = {
                    'policy': policy[:sample_size],
                    'value': value[:sample_size]
                }

                # Log des métriques de gradients
                monitor.log_gradient_metrics(
                    epoch=i,
                    model=model,
                    sample_inputs=sample_inputs,
                    sample_targets=sample_targets
                )

            except Exception as e:
                print(f"⚠️  Erreur lors de l'analyse des gradients: {e}")

    # ============================================================================
    # FINALISATION
    # ============================================================================

    total_time = time.time() - start_time

    # Statistiques finales d'augmentation
    if config['augmentation']['use_augmentation']:
        total_samples = config['train']['epochs'] * config['train']['N'] * config['augmentation']['nb_rotation']
        print(f"\n📊 Statistiques d'augmentation:")
        print(f"   🔄 {config['augmentation']['nb_rotation']} transformations par époque")
        print(f"   📈 Échantillons par époque: {config['train']['N']} → {config['train']['N'] * config['augmentation']['nb_rotation']}")
        print(f"   🎯 Total d'échantillons traités: {total_samples:,}")
        print(f"   ⏱️  Temps total: {total_time:.2f}s")

    # Log du résumé final
    final_metrics = metrics  # Dernières métriques d'entraînement
    monitor.log_final_summary(total_time, final_metrics)

    # Obtenir les statistiques de résumé
    summary_stats = monitor.get_summary_stats()
    print(f"\n🏆 Résumé de l'entraînement:")
    print(f"   📊 Accuracy finale: {summary_stats.get('final_policy_accuracy', 0):.4f}")
    print(f"   📊 Meilleure accuracy: {summary_stats.get('best_policy_accuracy', 0):.4f}")
    print(f"   📊 MSE finale: {summary_stats.get('final_value_mse', 0):.4f}")
    print(f"   📊 Santé des gradients: {summary_stats.get('gradient_health', 'UNKNOWN')}")

    # Terminer la session WandB
    monitor.finish()

    # Retourner les résultats pour compatibilité
    return (
        val if 'val' in locals() else None,
        pd.DataFrame(all_history),
        pd.DataFrame(val_loss_history),
        total_time,
        logger.lrs
    )