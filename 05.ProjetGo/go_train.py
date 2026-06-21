import time
import pandas as pd
import gc
import golois
import numpy as np
from typing import Dict, Any, Optional, Tuple

from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from training_monitor import TrainingMonitor

from go_callbacks import LrLogger


class GoTrainer:
    """
    Classe principale pour l'entraînement des modèles Go - Version avec deux modes d'augmentation

    MODES D'AUGMENTATION DISPONIBLES:

    Mode 1 "random_per_batch" (mode actuel):
    - getBatch() → applique 1 transformation aléatoire (T1-T7) sur tout le batch → train
    - Rapide, efficace, diversité par époque

    Mode 2 "all_transforms_per_batch":
    - getBatch() → applique les 8 transformations (T0-T7) → train sur chaque transformation
    - Plus de données par batch, apprentissage plus exhaustif
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le trainer avec la configuration

        Args:
            config: Configuration d'entraînement avec section 'augmentation':
                {
                    'augmentation': {
                        'use_augmentation': bool,
                        'mode': 'random_per_batch' | 'all_transforms_per_batch',
                        'nb_rotation': int (toujours 8),
                        'transform_probability': float (pour mode random_per_batch)
                    }
                }
        """
        self.config = config
        self.monitor = None
        self.logger = LrLogger()
        self.start_time = None

        # Historiques d'entraînement
        self.all_history = []
        self.val_loss_history = []

        # Configuration de l'augmentation
        aug_config = config['augmentation']
        self.use_augmentation = aug_config.get('use_augmentation', False)
        self.augmentation_mode = aug_config.get('mode', 'random_per_batch')
        self.nb_rotation = aug_config.get('nb_rotation', 8)
        self.transform_probability = aug_config.get('transform_probability', 0.8)

        # Validation du mode d'augmentation
        valid_modes = ['random_per_batch', 'all_transforms_per_batch']
        if self.augmentation_mode not in valid_modes:
            raise ValueError(f"Mode d'augmentation invalide: {self.augmentation_mode}. "
                             f"Modes valides: {valid_modes}")

        # Noms des transformations pour l'affichage
        self.transform_names = [
            "Original", "Rot90°", "Rot180°", "Rot270°",
            "MirrorH", "MirrorV", "DiagMain", "DiagAnti"
        ]

        print(f"🔧 Configuration augmentation:")
        print(f"   📊 Activée: {self.use_augmentation}")
        if self.use_augmentation:
            print(f"   🎯 Mode: {self.augmentation_mode}")
            if self.augmentation_mode == 'random_per_batch':
                print(f"   🎲 Probabilité transformation: {self.transform_probability:.0%}")
            elif self.augmentation_mode == 'all_transforms_per_batch':
                print(f"   🔄 Transformations par batch: {self.nb_rotation}")

    def _setup_monitoring(self, model):
        """Configure le monitoring avec TrainingMonitor"""
        self.monitor = TrainingMonitor(
            project_name="Go-Project",
            entity=self.config.get('entity', 'Emiasd'),
            experiment_name=self.config.get('experiment_name', 'go-experiment')
        )

        # Initialiser WandB avec la config adaptée
        monitoring_config = self.config.copy()
        monitoring_config['augmentation_mode'] = self.augmentation_mode
        monitoring_config['transform_probability'] = self.transform_probability
        monitoring_config['nb_rotation'] = self.nb_rotation

        mode_tag = self.augmentation_mode if self.use_augmentation else "no-augmentation"

        self.monitor.initialize_wandb(
            config=monitoring_config,
            tags=[
                "training", "go",
                self.config.get('model_type', 'mobilenet'),
                mode_tag
            ],
            notes=f"Go training - {self.config.get('experiment_name', '')} - Mode: {mode_tag}"
        )

        # Log des informations du modèle
        self.monitor.log_model_info(model)

    def _prepare_initial_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prépare les structures de données initiales"""
        planes = 31
        moves = 361
        N = self.config['train']['N']

        # Génération des structures de données
        input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
        policy = keras.utils.to_categorical(np.random.randint(moves, size=(N,)))
        value = np.random.randint(2, size=(N,)).astype('float32')
        end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
        groups = np.zeros((N, 19, 19, 1)).astype('float32')

        # Récupération des données de validation initiales
        print("📊 Récupération des données de validation initiales...", flush=True)
        golois.getValidation(input_data, policy, value, end)

        return input_data, policy, value, end, groups

    def _get_optimizer(self) -> optimizers.Optimizer:
        """Configure l'optimiseur avec le scheduler de learning rate"""
        train_config = self.config['train']

        # Calcul des batches par époque en fonction du mode d'augmentation
        base_batches_per_epoch = train_config['N'] // train_config['batch']

        if self.use_augmentation and self.augmentation_mode == 'all_transforms_per_batch':
            # Mode exhaustif: x8 plus de batches par époque
            batches_per_epoch = base_batches_per_epoch * self.nb_rotation
            print(
                f"📈 Mode all_transforms: {base_batches_per_epoch} * {self.nb_rotation} = {batches_per_epoch} batches/époque")
        else:
            # Mode random ou pas d'augmentation: même nombre de batches
            batches_per_epoch = base_batches_per_epoch
            print(f"📈 Mode standard: {batches_per_epoch} batches/époque")

        total_epochs = train_config['epochs']
        total_batches = batches_per_epoch * total_epochs
        first_decay_steps = total_batches // train_config.get('nb_cosinedecay_cycle', 1)

        print(f"📈 Scheduler LR: {total_epochs} époques, {total_batches} batches, {first_decay_steps} steps par cycle")

        # Configuration du scheduler cosine
        cosine_lr = CosineDecayRestarts(
            initial_learning_rate=train_config['initial_lr'],
            first_decay_steps=first_decay_steps,
            t_mul=train_config.get('t_mul_lr', 1.0),
            m_mul=train_config.get('m_mul_lr', 1.0),
            alpha=train_config.get('alpha_lr', 0.0)
        )

        # Choix de l'optimiseur
        if train_config['initial_lr'] != train_config.get('alpha_lr', 0.0):
            optimizer = optimizers.AdamW(learning_rate=cosine_lr, clipnorm=1.0)
        else:
            optimizer = optimizers.legacy.Adam(
                learning_rate=train_config['initial_lr'],
                clipnorm=1.0
            )

        return optimizer

    def _apply_go_transformation(self, board, policy, transform_id):
        """Applique une transformation spécifique au plateau et à la politique"""
        if transform_id == 0:
            return  # Identité

        elif transform_id == 1:  # Rotation 90°
            board[:] = np.rot90(board, k=-1, axes=(0, 1))
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.rot90(policy_2d, k=-1, axes=(0, 1))
                policy[:] = policy_2d.flatten()

        elif transform_id == 2:  # Rotation 180°
            board[:] = np.rot90(board, k=2, axes=(0, 1))
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.rot90(policy_2d, k=2, axes=(0, 1))
                policy[:] = policy_2d.flatten()

        elif transform_id == 3:  # Rotation 270°
            board[:] = np.rot90(board, k=1, axes=(0, 1))
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.rot90(policy_2d, k=1, axes=(0, 1))
                policy[:] = policy_2d.flatten()

        elif transform_id == 4:  # Miroir horizontal
            board[:] = np.flip(board, axis=1)
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.flip(policy_2d, axis=1)
                policy[:] = policy_2d.flatten()

        elif transform_id == 5:  # Miroir vertical
            board[:] = np.flip(board, axis=0)
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.flip(policy_2d, axis=0)
                policy[:] = policy_2d.flatten()

        elif transform_id == 6:  # Transpose
            board[:] = np.transpose(board, (1, 0, 2))
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.transpose(policy_2d, (1, 0))
                policy[:] = policy_2d.flatten()

        elif transform_id == 7:  # Anti-diagonal
            board[:] = np.flip(np.flip(np.transpose(board, (1, 0, 2)), axis=0), axis=1)
            if len(policy.shape) == 1 and len(policy) == 361:
                policy_2d = policy.reshape(19, 19)
                policy_2d[:] = np.flip(np.flip(np.transpose(policy_2d, (1, 0)), axis=0), axis=1)
                policy[:] = policy_2d.flatten()

    def _apply_random_augmentation(self, input_data, policy, value):
        """
        Mode 1: Applique une augmentation aléatoire à tout le batch (mode actuel)
        """
        N = input_data.shape[0]
        transform_stats = {i: 0 for i in range(8)}

        aug_start_time = time.time()

        if self.nb_rotation is None or self.nb_rotation == 1:
            transform_id = 0
        else:
            transform_id = np.random.randint(0, 8)  # Choix aléatoire entre T1-T7

        # Appliquer la même transformation à tout le batch
        if transform_id != 0 :
            for sample_idx in range(N):
                self._apply_go_transformation(
                    input_data[sample_idx],
                    policy[sample_idx],
                    transform_id
                )

        transform_stats[transform_id] = N
        aug_time = time.time() - aug_start_time

        return {
            'transform_stats': transform_stats,
            'augmentation_time': aug_time,
            'mode': 'random_per_batch',
            'transform_id': transform_id
        }

    def _apply_all_transforms_augmentation(self, model, input_data, policy, value, end, groups, epoch):
        """
        Mode 2: Applique toutes les transformations (T0-T7) sur le même batch
        """
        # Sauvegarder les données originales
        original_input = input_data.copy()
        original_policy = policy.copy()
        original_value = value.copy()

        epoch_histories = []
        aug_start_time = time.time()

        print(f"📊 Époque {epoch} - Mode exhaustif: entraînement sur {self.nb_rotation} transformations")

        for transform_id in range(self.nb_rotation):
            transform_start_time = time.time()

            # Restaurer les données originales
            input_data[:] = original_input
            policy[:] = original_policy
            value[:] = original_value

            # Appliquer la transformation transform_id à tout le batch
            if transform_id > 0:  # T0 = identité, pas de transformation
                for sample_idx in range(input_data.shape[0]):
                    self._apply_go_transformation(
                        input_data[sample_idx],
                        policy[sample_idx],
                        transform_id
                    )

            # Entraînement sur cette transformation
            history = model.fit(
                input_data,
                {'policy': policy, 'value': value},
                epochs=1,
                batch_size=self.config['train']['batch'],
                verbose=0 if transform_id > 0 else 1,  # Verbose seulement pour T0
                callbacks=[self.logger]
            )

            # Enregistrer les métriques
            transform_metrics = {}
            for key, val in history.history.items():
                transform_metrics[key] = val[0] if isinstance(val, list) and len(val) > 0 else val

            epoch_histories.append(transform_metrics)

            # Affichage pour chaque transformation
            transform_time = time.time() - transform_start_time
            print(f"  🔀 {self.transform_names[transform_id]:8s} (T{transform_id}): "
                  f"loss={transform_metrics['loss']:.4f}, "
                  f"policy_acc={transform_metrics['policy_categorical_accuracy']:.4f}, "
                  f"value_mse={transform_metrics['value_mse']:.4f}, "
                  f"time={transform_time:.1f}s")

        # Calculer les métriques moyennes de l'époque
        avg_metrics = {}
        for key in epoch_histories[0].keys():
            avg_metrics[key] = np.mean([h[key] for h in epoch_histories])

        avg_metrics['epoch'] = epoch
        total_aug_time = time.time() - aug_start_time

        # Statistiques d'augmentation
        transform_stats = {i: input_data.shape[0] for i in range(self.nb_rotation)}
        for i in range(self.nb_rotation, 8):
            transform_stats[i] = 0

        aug_stats = {
            'transform_stats': transform_stats,
            'augmentation_time': total_aug_time,
            'mode': 'all_transforms_per_batch',
            'nb_transforms_applied': self.nb_rotation,
            'epoch_histories': epoch_histories
        }

        return avg_metrics, aug_stats

    def _filter_numeric_metrics(self, metrics: Dict) -> Dict:
        """Filtre les métriques pour ne garder que les valeurs numériques"""
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_metrics[key] = value
            elif isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                numeric_metrics[key] = float(value)
        return numeric_metrics

    def _train_single_epoch_random_mode(self, model, input_data, policy, value, end, groups, epoch: int):
        """
        Mode 1: Entraîne une époque avec augmentation aléatoire par batch
        """
        # Charger les nouvelles données pour cette époque
        golois.getBatch(input_data, policy, value, end, groups, epoch * self.config['train']['N'])

        # Appliquer l'augmentation aléatoire
        aug_stats = self._apply_random_augmentation(input_data, policy, value)

        # Affichage simple
        print(f"✅ Époque {epoch} - Mode aléatoire: T{aug_stats['transform_id']} "
              f"({self.transform_names[aug_stats['transform_id']]}) "
              f"({aug_stats['augmentation_time']:.2f}s)")

        # Entraînement
        history = model.fit(
            input_data,
            {'policy': policy, 'value': value},
            epochs=1,
            batch_size=self.config['train']['batch'],
            verbose=1,
            callbacks=[self.logger]
        )

        # Préparer les métriques
        metrics = {}
        for key, val in history.history.items():
            metrics[key] = val[0] if isinstance(val, list) and len(val) > 0 else val

        metrics['epoch'] = epoch
        metrics['augmentation_time'] = aug_stats['augmentation_time']
        metrics['transform_id'] = aug_stats['transform_id']

        return metrics, aug_stats

    def _train_single_epoch_all_transforms_mode(self, model, input_data, policy, value, end, groups, epoch: int):
        """
        Mode 2: Entraîne une époque avec toutes les transformations
        """
        # Charger les nouvelles données pour cette époque
        golois.getBatch(input_data, policy, value, end, groups, epoch * self.config['train']['N'])

        # Appliquer toutes les transformations
        metrics, aug_stats = self._apply_all_transforms_augmentation(
            model, input_data, policy, value, end, groups, epoch
        )

        print(f"✅ Époque {epoch} - Mode exhaustif: {self.nb_rotation} transformations "
              f"({aug_stats['augmentation_time']:.2f}s)")

        return metrics, aug_stats

    def _train_single_epoch_no_augmentation(self, model, input_data, policy, value, end, groups, epoch: int):
        """
        Mode sans augmentation
        """
        # Charger les nouvelles données pour cette époque
        golois.getBatch(input_data, policy, value, end, groups, epoch * self.config['train']['N'])

        print(f"✅ Époque {epoch} - Sans augmentation")

        # Entraînement direct
        history = model.fit(
            input_data,
            {'policy': policy, 'value': value},
            epochs=1,
            batch_size=self.config['train']['batch'],
            verbose=1,
            callbacks=[self.logger]
        )

        # Préparer les métriques
        metrics = {}
        for key, val in history.history.items():
            metrics[key] = val[0] if isinstance(val, list) and len(val) > 0 else val

        metrics['epoch'] = epoch

        return metrics, None

    def _should_validate(self, epoch: int) -> bool:
        """Détermine si une validation doit être effectuée à cette époque"""
        train_config = self.config['train']
        validation_frequency = train_config.get('validation_frequency', 10)

        return (epoch == 1 or
                epoch % validation_frequency == 0 or
                epoch == train_config['epochs'])

    def _validate_model(self, model, input_data, policy, value, end, epoch):
        """Effectue la validation du modèle"""
        print(f"\n🔍 Validation à l'époque {epoch}...")

        # Récupérer des données de validation fraîches
        golois.getValidation(input_data, policy, value, end)

        val = model.evaluate(
            input_data,
            [policy, value],
            verbose=1,
            batch_size=self.config['train']['batch']
        )

        val_metrics = {
            'epoch': epoch,
            'val_loss': val[0],
            'val_policy_loss': val[1],
            'val_value_loss': val[2],
            'val_policy_categorical_accuracy': val[3],
            'val_value_mse': val[4]
        }

        self.val_loss_history.append(val_metrics)

        print(f"📈 Validation Époque {epoch}: "
              f"loss={val[0]:.4f}, "
              f"policy_acc={val[3]:.4f}, "
              f"value_mse={val[4]:.4f}")

        return val_metrics, val

    def train(self, model) -> Tuple[Any, pd.DataFrame, pd.DataFrame, float, list]:
        """
        Méthode principale d'entraînement avec modes d'augmentation configurables

        Args:
            model: Modèle Keras à entraîner

        Returns:
            Tuple contenant:
            - val: Résultats de validation finale
            - all_history: DataFrame avec l'historique d'entraînement
            - val_loss_history: DataFrame avec l'historique de validation
            - total_time: Temps total d'entraînement
            - lrs: Liste des learning rates
        """
        self.start_time = time.time()

        # Setup du monitoring
        self._setup_monitoring(model)

        # Configuration d'entraînement
        train_config = self.config['train']

        # Préparation des données initiales
        input_data, policy, value, end, groups = self._prepare_initial_data()

        # Configuration du modèle
        optimizer = self._get_optimizer()
        model.compile(
            optimizer=optimizer,
            loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
            loss_weights={'policy': train_config['policy_weight'], 'value': train_config['value_weight']},
            metrics={'policy': 'categorical_accuracy', 'value': 'mse'}
        )

        # Variables pour validation
        val = None
        total_epochs = train_config['epochs']

        # 🚀 BOUCLE D'ENTRAÎNEMENT PRINCIPALE
        for epoch in range(1, total_epochs + 1):
            epoch_start_time = time.time()

            # Choisir la méthode d'entraînement selon le mode
            if not self.use_augmentation:
                metrics, aug_stats = self._train_single_epoch_no_augmentation(
                    model, input_data, policy, value, end, groups, epoch
                )
            elif self.augmentation_mode == 'random_per_batch':
                metrics, aug_stats = self._train_single_epoch_random_mode(
                    model, input_data, policy, value, end, groups, epoch
                )
            elif self.augmentation_mode == 'all_transforms_per_batch':
                metrics, aug_stats = self._train_single_epoch_all_transforms_mode(
                    model, input_data, policy, value, end, groups, epoch
                )
            else:
                raise ValueError(f"Mode d'augmentation non supporté: {self.augmentation_mode}")

            # Ajouter les métriques à l'historique
            self.all_history.append(metrics)

            # Calcul du temps d'époque
            epoch_time = time.time() - epoch_start_time
            timing_metrics = {
                "epoch_duration": epoch_time,
                "total_elapsed": time.time() - self.start_time
            }

            # Logging vers le monitor
            current_lr = self.logger.lrs[-1] if self.logger.lrs else None
            log_metrics = self._filter_numeric_metrics(metrics)

            # Ajouter les stats d'augmentation aux métriques de log
            if aug_stats:
                for transform_id, count in aug_stats['transform_stats'].items():
                    log_metrics[f'transform_{transform_id}_count'] = count
                log_metrics['augmentation_mode'] = aug_stats.get('mode', 'none')

            self.monitor.log_epoch_metrics(
                epoch=epoch,
                train_metrics=log_metrics,
                learning_rate=current_lr,
                timing=self._filter_numeric_metrics(timing_metrics)
            )

            # Nettoyage mémoire périodique
            if epoch % 5 == 0:
                gc.collect()

            # Validation conditionnelle
            if self._should_validate(epoch):
                val_metrics, val = self._validate_model(model, input_data, policy, value, end, epoch)

                # Log validation vers le monitor
                self.monitor.log_epoch_metrics(
                    epoch=epoch,
                    train_metrics=log_metrics,
                    val_metrics=self._filter_numeric_metrics(val_metrics),
                    learning_rate=current_lr,
                    timing=self._filter_numeric_metrics(timing_metrics)
                )

                # Sauvegarde du meilleur modèle
                self.monitor.log_model_checkpoint(
                    model=model,
                    epoch=epoch,
                    metrics=val_metrics,
                    save_condition=True
                )

            # Monitoring des gradients (moins fréquent)
            if epoch % 20 == 0:
                try:
                    sample_size = min(32, train_config['batch'])
                    sample_inputs = input_data[:sample_size]
                    sample_targets = {
                        'policy': policy[:sample_size],
                        'value': value[:sample_size]
                    }

                    self.monitor.log_gradient_metrics(
                        epoch=epoch,
                        model=model,
                        sample_inputs=sample_inputs,
                        sample_targets=sample_targets
                    )
                except Exception as e:
                    print(f"⚠️  Erreur analyse gradients: {e}")

        # Finalisation
        total_time = time.time() - self.start_time

        # Statistiques finales
        print(f"\n📊 Statistiques finales:")
        print(f"   🎯 {total_epochs} époques")
        print(f"   🔧 Mode augmentation: {self.augmentation_mode if self.use_augmentation else 'Désactivée'}")
        print(f"   ⏱️  Temps total: {total_time:.2f}s ({total_time / total_epochs:.2f}s/époque)")

        if self.use_augmentation:
            total_samples = total_epochs * train_config['N']
            if self.augmentation_mode == 'all_transforms_per_batch':
                total_samples *= self.nb_rotation
                print(f"   🔄 Total échantillons traités: {total_samples:,} ({self.nb_rotation}x augmentation)")
            else:
                print(f"   🎲 Total échantillons traités: {total_samples:,} (augmentation aléatoire)")
            print(f"   📈 Vitesse: {total_samples / total_time:.0f} échantillons/seconde")

        # Log du résumé final
        final_metrics = self.all_history[-1] if self.all_history else {}
        self.monitor.log_final_summary(total_time, final_metrics)

        # Obtenir les statistiques de résumé
        summary_stats = self.monitor.get_summary_stats()
        print(f"\n🏆 Résumé de l'entraînement:")
        print(f"   📊 Accuracy finale: {summary_stats.get('final_policy_accuracy', 0):.4f}")
        print(f"   📊 Meilleure accuracy: {summary_stats.get('best_policy_accuracy', 0):.4f}")
        print(f"   📊 MSE finale: {summary_stats.get('final_value_mse', 0):.4f}")

        # Terminer la session
        self.monitor.finish()

        return (
            val,
            pd.DataFrame(self.all_history),
            pd.DataFrame(self.val_loss_history),
            total_time,
            self.logger.lrs
        )