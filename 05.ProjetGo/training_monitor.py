# ============================================================================
# CLASSE UNIFIÉE DE MONITORING POUR LE PROJET GO MOBILENET - VERSION CORRIGÉE
# Fichier: training_monitor.py
# ============================================================================

import wandb
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Any
import time
from datetime import datetime


class TrainingMonitor:
    """
    Classe pour le monitoring de l'entraînement avec WandB.
    Version corrigée pour gérer les valeurs non-numériques.
    """

    def __init__(self, project_name: str = None,
                 entity: Optional[str] = None,
                 experiment_name: Optional[str] = None):

        self.project_name = project_name
        self.entity = entity
        self.experiment_name = self._generate_experiment_name(experiment_name)
        self.run = None

        # Historique des métriques
        self.metrics_history = []
        self.gradient_history = []

        # État de l'entraînement
        self.best_val_loss = float('inf')
        self.start_time = None

        # Configuration des seuils d'alertes
        self.thresholds = {
            'policy_accuracy_min': 0.35,  # Accuracy minimale acceptable
            'value_mse_max': 0.10,  # MSE maximale acceptable
            'gradient_norm_min': 1e-6,  # Gradient vanishing
            'gradient_norm_max': 10.0,  # Gradient explosion
            'loss_explosion_factor': 5.0  # Facteur d'explosion de loss
        }

    def _generate_experiment_name(self, experiment_name) -> str:
        """Génère un nom d'expérience automatique"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{experiment_name}_{timestamp}"

    def _safe_convert_to_float(self, value, key="unknown"):
        """
        Convertit une valeur en float de manière sécurisée

        Args:
            value: Valeur à convertir
            key: Nom de la clé pour le debug

        Returns:
            float ou None si la conversion échoue
        """
        try:
            # Si c'est déjà un nombre
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)

            # Si c'est une liste, prendre le premier élément
            if isinstance(value, list) and len(value) > 0:
                return self._safe_convert_to_float(value[0], key)

            # Si c'est un tensor/array numpy
            if hasattr(value, 'numpy'):
                return float(value.numpy())
            if isinstance(value, np.ndarray):
                return float(value.item() if value.size == 1 else value.mean())

            # Si c'est une string qui représente un nombre
            if isinstance(value, str):
                # Vérifier si c'est un nombre en string
                try:
                    return float(value)
                except ValueError:
                    # C'est une vraie string, on la ignore pour WandB
                    #print(f"⚠️  Ignoring non-numeric value for key '{key}': '{value}'")
                    return None

            # Autres cas
            print(f"⚠️  Cannot convert value for key '{key}': {type(value)} = {value}")
            return None

        except Exception as e:
            print(f"⚠️  Error converting value for key '{key}': {e}")
            return None

    def initialize_wandb(self, config: Dict[str, Any],
                         tags: Optional[List[str]] = None,
                         notes: Optional[str] = None):
        """
        Initialise WandB avec la configuration
        """
        # Configuration enrichie
        default_config = {
            # Dataset
            "input_shape": [19, 19, 31],
            "num_classes": 361,
            # Framework
            "framework": f"TensorFlow/{tf.__version__}",
            "experiment_name": self.experiment_name
        }

        # Fusion avec la config utilisateur
        wandb_config = {**default_config, **config}

        # Tags par défaut
        default_tags = ["go", "mobilenet", "neural-network"]
        if tags:
            default_tags.extend(tags)

        # Initialiser WandB
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=wandb_config,
            tags=default_tags,
            notes=notes or f"Training for Go - {self.experiment_name}",
            reinit=True
        )

        self.start_time = time.time()
        print(f"🚀 WandB run initialisée: {self.run.url}")
        return self.run

    def log_model_info(self, model):
        """Log des informations du modèle"""
        if self.run:
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/non_trainable_parameters": total_params - trainable_params,
                "model/layers_count": len(model.layers)
            })

    def log_epoch_metrics(self, epoch: int,
                          train_metrics: Dict[str, float],
                          val_metrics: Optional[Dict[str, float]] = None,
                          learning_rate: Optional[float] = None,
                          timing: Optional[Dict[str, float]] = None):
        """
        Log des métriques d'une époque avec gestion sécurisée des types
        """
        if not self.run:
            return

        # Préparer les métriques pour WandB
        wandb_metrics = {"epoch": epoch}

        # Métriques d'entraînement
        for key, value in train_metrics.items():
            converted_value = self._safe_convert_to_float(value, f"train/{key}")
            if converted_value is not None:
                wandb_metrics[f"train/{key}"] = converted_value

        # Métriques de validation
        if val_metrics:
            for key, value in val_metrics.items():
                clean_key = key.replace('val_', '') if key.startswith('val_') else key
                converted_value = self._safe_convert_to_float(value, f"validation/{clean_key}")
                if converted_value is not None:
                    wandb_metrics[f"validation/{clean_key}"] = converted_value

        # Learning rate
        if learning_rate is not None:
            converted_lr = self._safe_convert_to_float(learning_rate, "learning_rate")
            if converted_lr is not None:
                wandb_metrics["optimization/learning_rate"] = converted_lr

        # Timing
        if timing:
            for key, value in timing.items():
                converted_value = self._safe_convert_to_float(value, f"timing/{key}")
                if converted_value is not None:
                    wandb_metrics[f"timing/{key}"] = converted_value

        # Métriques système
        if self.start_time:
            wandb_metrics["timing/total_elapsed"] = time.time() - self.start_time

        # Log vers WandB (seulement les valeurs numériques)
        if len(wandb_metrics) > 1:  # Plus que juste l'epoch
            wandb.log(wandb_metrics, step=epoch)

        # Stocker dans l'historique (toutes les valeurs, même non-numériques)
        metrics_entry = {
            'epoch': epoch,
            'train': train_metrics,
            'validation': val_metrics or {},
            'learning_rate': learning_rate,
            'timing': timing or {}
        }
        self.metrics_history.append(metrics_entry)

    def log_gradient_metrics(self, epoch: int, model, sample_inputs, sample_targets):
        """
        Log des métriques de gradients simplifiées
        """
        if not self.run:
            return

        try:
            # Calcul des gradients
            with tf.GradientTape() as tape:
                predictions = model(sample_inputs, training=True)

                if isinstance(predictions, dict):
                    policy_pred = predictions['policy']
                    value_pred = predictions['value']
                else:
                    policy_pred, value_pred = predictions

                # Calcul simple des losses
                policy_loss = tf.keras.losses.categorical_crossentropy(
                    sample_targets['policy'], policy_pred
                )
                value_loss = tf.keras.losses.mse(
                    sample_targets['value'], tf.squeeze(value_pred)
                )

                total_loss = tf.reduce_mean(policy_loss) + tf.reduce_mean(value_loss)

            gradients = tape.gradient(total_loss, model.trainable_variables)

            # Calculer les normes de gradients
            gradient_norms = []
            for grad in gradients:
                if grad is not None:
                    gradient_norms.append(tf.norm(grad).numpy())

            if gradient_norms:
                # Métriques de gradients
                gradient_metrics = {
                    "gradients/mean_norm": np.mean(gradient_norms),
                    "gradients/max_norm": np.max(gradient_norms),
                    "gradients/min_norm": np.min(gradient_norms),
                    "gradients/std_norm": np.std(gradient_norms)
                }

                # Détection de problèmes
                mean_norm = gradient_metrics["gradients/mean_norm"]
                max_norm = gradient_metrics["gradients/max_norm"]

                if mean_norm < self.thresholds['gradient_norm_min']:
                    gradient_metrics["alerts/gradient_vanishing"] = 1
                    print(f"⚠️  Gradient vanishing détecté à l'époque {epoch} (norme: {mean_norm:.2e})")

                if max_norm > self.thresholds['gradient_norm_max']:
                    gradient_metrics["alerts/gradient_explosion"] = 1
                    print(f"🚨 Gradient explosion détecté à l'époque {epoch} (norme: {max_norm:.2e})")

                # Log vers WandB
                wandb.log(gradient_metrics, step=epoch)

                # Stocker dans l'historique
                self.gradient_history.append({
                    'epoch': epoch,
                    'mean_norm': mean_norm,
                    'max_norm': max_norm,
                    'min_norm': gradient_metrics["gradients/min_norm"]
                })

        except Exception as e:
            print(f"⚠️  Erreur lors du calcul des gradients: {e}")

    def log_model_checkpoint(self, model, epoch: int, metrics: Dict[str, float],
                             save_condition: bool = False):
        """
        Sauvegarde conditionnelle du modèle
        """
        if not self.run or not save_condition:
            return

        # Évaluer si c'est le meilleur modèle
        current_val_loss = metrics.get('val_total_loss', metrics.get('val_loss', float('inf')))

        # Conversion sécurisée
        converted_loss = self._safe_convert_to_float(current_val_loss, "val_loss")
        if converted_loss is None:
            print(f"⚠️  Cannot save model: invalid validation loss value")
            return

        current_val_loss = converted_loss

        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            # Sauvegarder le modèle
            model_filename = f"best_model_epoch_{epoch}.h5"
            model.save(model_filename)

            # Obtenir les métriques avec conversion sécurisée
            policy_acc = self._safe_convert_to_float(
                metrics.get('val_policy_categorical_accuracy', 0),
                "policy_accuracy"
            ) or 0

            value_mse = self._safe_convert_to_float(
                metrics.get('val_value_mse', float('inf')),
                "value_mse"
            ) or float('inf')

            # Créer un artefact WandB
            artifact = wandb.Artifact(
                name=f"go_mobilenet_best_epoch_{epoch}",
                type="model",
                description=f"Best Go model at epoch {epoch}",
                metadata={
                    "epoch": epoch,
                    "validation_loss": float(current_val_loss),
                    "policy_accuracy": float(policy_acc),
                    "value_mse": float(value_mse)
                }
            )

            artifact.add_file(model_filename)
            self.run.log_artifact(artifact)

            # Log de la métrique
            wandb.log({"model/best_val_loss": float(current_val_loss)}, step=epoch)

            print(f"💾 Nouveau meilleur modèle sauvé: epoch {epoch}, val_loss: {current_val_loss:.4f}")

            # Nettoyer le fichier local
            try:
                import os
                os.remove(model_filename)
            except:
                pass

    def log_final_summary(self, total_time: float, final_metrics: Dict):
        """
        Log du résumé final de l'entraînement
        """
        if not self.run:
            return

        # Conversion sécurisée des métriques finales
        final_policy_acc = self._safe_convert_to_float(
            final_metrics.get('policy_categorical_accuracy', 0),
            "final_policy_accuracy"
        ) or 0

        final_value_mse = self._safe_convert_to_float(
            final_metrics.get('value_mse', float('inf')),
            "final_value_mse"
        ) or float('inf')

        # Métriques finales
        final_summary = {
            "final/total_training_time": total_time,
            "final/best_validation_loss": self.best_val_loss,
            "final/epochs_completed": len(self.metrics_history),
            "final/final_policy_accuracy": final_policy_acc,
            "final/final_value_mse": final_value_mse
        }

        # Statistiques de l'entraînement
        if self.metrics_history:
            policy_accuracies = []
            for h in self.metrics_history:
                acc = h['train'].get('policy_categorical_accuracy', 0)
                converted_acc = self._safe_convert_to_float(acc, "policy_accuracy")
                if converted_acc is not None:
                    policy_accuracies.append(converted_acc)

            if policy_accuracies:
                final_summary.update({
                    "stats/max_policy_accuracy": max(policy_accuracies),
                    "stats/mean_policy_accuracy": np.mean(policy_accuracies),
                    "stats/final_policy_accuracy": policy_accuracies[-1]
                })

        # Statistiques des gradients
        if self.gradient_history:
            mean_norms = [h['mean_norm'] for h in self.gradient_history]
            final_summary.update({
                "stats/gradient_mean_norm": np.mean(mean_norms),
                "stats/gradient_final_norm": mean_norms[-1] if mean_norms else 0
            })

        # Log final
        wandb.log(final_summary)

        print(f"\n🏆 Entraînement terminé:")
        print(f"   Temps total: {total_time:.2f}s")
        print(f"   Meilleure validation loss: {self.best_val_loss:.4f}")
        print(f"   Accuracy finale: {final_summary.get('final/final_policy_accuracy', 0):.4f}")

    def finish(self):
        """
        Termine la session WandB
        """
        if self.run:
            wandb.finish()
            print("✅ Session WandB terminée")

    def get_summary_stats(self) -> Dict:
        """
        Retourne un résumé des statistiques d'entraînement
        """
        if not self.metrics_history:
            return {}

        # Extraire les métriques d'entraînement avec conversion sécurisée
        policy_accuracies = []
        value_mses = []
        losses = []

        for entry in self.metrics_history:
            train_metrics = entry['train']

            # Policy accuracy
            policy_acc = self._safe_convert_to_float(
                train_metrics.get('policy_categorical_accuracy', 0),
                "policy_accuracy"
            )
            if policy_acc is not None:
                policy_accuracies.append(policy_acc)

            # Value MSE
            value_mse = self._safe_convert_to_float(
                train_metrics.get('value_mse', float('inf')),
                "value_mse"
            )
            if value_mse is not None:
                value_mses.append(value_mse)

            # Loss
            loss = self._safe_convert_to_float(
                train_metrics.get('loss', float('inf')),
                "loss"
            )
            if loss is not None:
                losses.append(loss)

        return {
            'total_epochs': len(self.metrics_history),
            'best_policy_accuracy': max(policy_accuracies) if policy_accuracies else 0,
            'final_policy_accuracy': policy_accuracies[-1] if policy_accuracies else 0,
            'best_value_mse': min(value_mses) if value_mses else float('inf'),
            'final_value_mse': value_mses[-1] if value_mses else float('inf'),
            'best_validation_loss': self.best_val_loss,
            'gradient_health': 'HEALTHY' if not self.gradient_history or
                                            all(h['mean_norm'] > self.thresholds['gradient_norm_min'] for h in
                                                self.gradient_history[-5:])
            else 'WARNING'
        }