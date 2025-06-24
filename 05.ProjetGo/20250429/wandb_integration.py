# ============================================================================
# INTÉGRATION WEIGHTS & BIASES POUR LE PROJET GO MOBILENET
# Fichier: wandb_integration.py
# ============================================================================

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import os
import json
from datetime import datetime
from tensorflow.keras.callbacks import Callback
import tensorflow as tf


# ============================================================================
# 1. TRACKER PRINCIPAL WANDB
# ============================================================================

class WandbGoTracker:
    """
    Intégration complète de Weights & Biases pour le projet Go MobileNet
    """

    def __init__(self, project_name: str = "go-mobilenet",
                 entity: Optional[str] = None,
                 experiment_name: Optional[str] = None):
        """
        Initialise le tracker WandB

        Args:
            project_name: Nom du projet WandB
            entity: Nom de l'équipe/utilisateur WandB (optionnel)
            experiment_name: Nom de l'expérience (généré auto si None)
        """
        self.project_name = project_name
        self.entity = entity
        self.experiment_name = experiment_name or self._generate_experiment_name()
        self.run = None

    def _generate_experiment_name(self) -> str:
        """Génère un nom d'expérience automatique"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"mobilenet_experiment_{timestamp}"

    def initialize_run(self, config: Dict[str, Any],
                       tags: Optional[List[str]] = None,
                       notes: Optional[str] = None):
        """
        Initialise une run WandB avec la configuration

        Args:
            config: Configuration de l'entraînement
            tags: Tags pour organiser les expériences
            notes: Notes descriptives
        """

        # Configuration par défaut enrichie
        default_config = {
            # Hyperparamètres du modèle
            "model_type": "MobileNet",
            "block_num": 8,
            "filters": 31,
            "expansion_factor": 4,
            "squeeze_excitation": True,
            "activation": "swish",
            "dropout_rate": 0.2,

            # Hyperparamètres d'entraînement
            "epochs": 500,
            "batch_size": 32,
            "initial_lr": 0.02,
            "lr_schedule": "cosine_annealing",
            "cosine_cycles": 1,
            "policy_weight": 1.0,
            "value_weight": 1.0,
            "optimizer": "AdamW",
            "clipnorm": 1.0,

            # Configuration des données
            "dataset_size": 10000,
            "input_shape": [19, 19, 31],
            "num_classes": 361,
            "data_augmentation": "8-fold_symmetry",

            # Métriques de performance
            "target_policy_accuracy": 0.55,
            "target_value_mse": 0.15,

            # Informations système
            "framework": "TensorFlow/Keras",
            "python_version": f"{tf.__version__}",
        }

        # Fusion avec la config utilisateur
        merged_config = {**default_config, **config}

        # Tags par défaut
        default_tags = ["go", "mobilenet", "neural-network", "reinforcement-learning"]
        if tags:
            default_tags.extend(tags)

        # Initialiser WandB
        self.run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=self.experiment_name,
            config=merged_config,
            tags=default_tags,
            notes=notes or f"MobileNet training for Go - {self.experiment_name}",
            reinit=True
        )

        print(f"🚀 WandB run initialisée: {self.run.url}")
        return self.run

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log des métriques à WandB"""
        if self.run:
            wandb.log(metrics, step=step)

    def log_model_architecture(self, model):
        """Log l'architecture du modèle"""
        if self.run:
            # Sauvegarder le summary du modèle
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            model_summary_text = '\n'.join(model_summary)

            # Log comme artefact
            with open("model_summary.txt", "w") as f:
                f.write(model_summary_text)

            artifact = wandb.Artifact("model_architecture", type="model_summary")
            artifact.add_file("model_summary.txt")
            self.run.log_artifact(artifact)

            # Log les paramètres du modèle
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

            wandb.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/non_trainable_parameters": total_params - trainable_params,
                "model/layers_count": len(model.layers)
            })

            # Nettoyer le fichier temporaire
            try:
                os.remove("model_summary.txt")
            except:
                pass

    def log_learning_rate_schedule(self, learning_rates: List[float]):
        """Log le schedule de learning rate"""
        if self.run and learning_rates:
            # Créer un graphique du LR schedule
            plt.figure(figsize=(10, 6))
            plt.plot(learning_rates)
            plt.title('Learning Rate Schedule')
            plt.xlabel('Batch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)

            # Log comme image
            wandb.log({"learning_rate_schedule": wandb.Image(plt)})
            plt.close()

    def log_training_diagnostics(self, analysis_results: Dict):
        """Log les résultats du diagnostic de convergence"""
        if not self.run:
            return

        # Métriques de santé
        health_mapping = {'HEALTHY': 4, 'WARNING': 3, 'CRITICAL': 2, 'UNKNOWN': 1}
        health_score = health_mapping.get(analysis_results.get('overall_health', 'UNKNOWN'), 1)

        wandb.log({
            "diagnostics/health_score": health_score,
            "diagnostics/health_status": analysis_results.get('overall_health', 'UNKNOWN'),
            "diagnostics/issues_count": len(analysis_results.get('issues', [])),
            "diagnostics/critical_issues": len(
                [i for i in analysis_results.get('issues', []) if i.get('severity') == 'CRITICAL']),
            "diagnostics/high_issues": len(
                [i for i in analysis_results.get('issues', []) if i.get('severity') == 'HIGH'])
        })

        # Métriques détaillées
        metrics_analysis = analysis_results.get('metrics_analysis', {})

        if 'policy' in metrics_analysis:
            policy = metrics_analysis['policy']
            wandb.log({
                "diagnostics/policy_current_accuracy": policy.get('current_accuracy', 0),
                "diagnostics/policy_max_accuracy": policy.get('max_accuracy', 0),
                "diagnostics/policy_improvement_rate": policy.get('improvement_rate', 0),
                "diagnostics/policy_plateau_detected": policy.get('plateau_detected', False),
                "diagnostics/policy_benchmark_status": policy.get('benchmark_comparison', 'UNKNOWN')
            })

        if 'value' in metrics_analysis:
            value = metrics_analysis['value']
            wandb.log({
                "diagnostics/value_current_mse": value.get('current_mse', float('inf')),
                "diagnostics/value_min_mse": value.get('min_mse', float('inf')),
                "diagnostics/value_convergence_rate": value.get('convergence_rate', 0),
                "diagnostics/value_stagnation_detected": value.get('stagnation_detected', False)
            })

    def log_model_checkpoint(self, model, epoch: int, metrics: Dict[str, float]):
        """Sauvegarde un checkpoint du modèle"""
        if not self.run:
            return

        # Sauvegarder le modèle
        model_filename = f"model_epoch_{epoch}.h5"
        model.save(model_filename)

        # Créer un artefact WandB
        artifact = wandb.Artifact(
            name=f"go_mobilenet_epoch_{epoch}",
            type="model",
            description=f"MobileNet model at epoch {epoch}",
            metadata={
                "epoch": epoch,
                "policy_accuracy": metrics.get('policy_categorical_accuracy', 0),
                "value_mse": metrics.get('value_mse', float('inf')),
                "total_loss": metrics.get('loss', float('inf'))
            }
        )

        artifact.add_file(model_filename)
        self.run.log_artifact(artifact)

        # Nettoyer le fichier local
        try:
            os.remove(model_filename)
        except:
            pass

        print(f"📦 Modèle sauvegardé sur WandB: epoch {epoch}")

    def finish_run(self):
        """Termine la run WandB"""
        if self.run:
            wandb.finish()
            print("✅ Run WandB terminée")


# ============================================================================
# 2. CALLBACK WANDB MANUEL
# ============================================================================

class WandbGoCallbackManual(Callback):
    """
    Callback WandB conçu pour être appelé manuellement
    Résout le problème epochs=1 en utilisant le vrai numéro d'époque
    """

    def __init__(self, wandb_tracker,
                 log_frequency: int = 1,
                 save_model_frequency: int = 20,
                 log_weights: bool = False):
        super().__init__()
        self.wandb_tracker = wandb_tracker
        self.log_frequency = log_frequency
        self.save_model_frequency = save_model_frequency
        self.log_weights = log_weights

    def on_epoch_end(self, epoch, logs=None):
        """
        Méthode appelée manuellement avec le VRAI numéro d'époque

        Args:
            epoch: Le vrai numéro d'époque (1, 2, 3, ...) - PAS celui de Keras
            logs: Dictionnaire des métriques
        """
        logs = logs or {}

        # Convertir epoch en base 1 pour plus de clarté
        true_epoch = epoch + 1 if epoch >= 0 else 1

        # Log des métriques de base
        if true_epoch % self.log_frequency == 0:

            metrics_to_log = {}

            # Training metrics
            for key, value in logs.items():
                if not key.startswith('val_'):
                    # S'assurer que c'est un float
                    val = float(value[0]) if isinstance(value, list) else float(value)
                    metrics_to_log[f"train/{key}"] = val
                else:
                    # Validation metrics (si présentes)
                    clean_key = key.replace('val_', '')
                    val = float(value[0]) if isinstance(value, list) else float(value)
                    metrics_to_log[f"validation/{clean_key}"] = val

            # Ajouter l'époque réelle
            metrics_to_log["system/epoch"] = true_epoch

            # Log vers WandB
            self.wandb_tracker.log_metrics(metrics_to_log, step=true_epoch)

            print(f"📊 WandB: Logged epoch {true_epoch} metrics")

        # Sauvegarde du modèle
        if true_epoch % self.save_model_frequency == 0:
            if hasattr(self, 'model') and self.model:
                self.wandb_tracker.log_model_checkpoint(
                    self.model,
                    true_epoch,
                    logs
                )
                print(f"💾 WandB: Model saved at epoch {true_epoch}")

        # Log des poids (optionnel)
        if (self.log_weights and
                true_epoch % (self.save_model_frequency * 2) == 0 and
                hasattr(self, 'model')):
            self._log_model_weights(true_epoch)

    def set_model(self, model):
        """Définit le modèle pour le callback (nécessaire pour appel manuel)"""
        self.model = model

    def log_validation_manually(self, epoch, val_metrics):
        """
        Log manuel des métriques de validation

        Args:
            epoch: Numéro d'époque (base 1)
            val_metrics: Dict des métriques de validation
        """
        metrics_to_log = {}

        for key, value in val_metrics.items():
            clean_key = key.replace('val_', '') if key.startswith('val_') else key
            metrics_to_log[f"validation/{clean_key}"] = float(value)

        self.wandb_tracker.log_metrics(metrics_to_log, step=epoch)
        print(f"✅ WandB: Logged validation metrics for epoch {epoch}")

    def log_learning_rate_manually(self, epoch, lr):
        """Log manuel du learning rate"""
        self.wandb_tracker.log_metrics({
            "optimization/learning_rate": float(lr)
        }, step=epoch)

    def log_timing_manually(self, epoch, timing_metrics):
        """Log manuel des métriques de timing"""
        timing_to_log = {}
        for key, value in timing_metrics.items():
            timing_to_log[f"timing/{key}"] = float(value)

        self.wandb_tracker.log_metrics(timing_to_log, step=epoch)

    def _log_model_weights(self, epoch):
        """Log les histogrammes des poids"""
        if not hasattr(self, 'model'):
            return

        weight_metrics = {}
        for layer in self.model.layers:
            if hasattr(layer, 'weights') and layer.weights:
                for weight in layer.weights:
                    weight_name = f"weights/{layer.name}_{weight.name.split(':')[0]}"
                    weight_metrics[weight_name] = wandb.Histogram(weight.numpy())

        if weight_metrics:
            wandb.log(weight_metrics, step=epoch)


# ============================================================================
# 3. GESTIONNAIRE D'EXPÉRIENCES ET SWEEPS
# ============================================================================

class WandbExperimentManager:
    """
    Gestionnaire pour les comparaisons d'expériences et sweeps
    """

    @staticmethod
    def create_hyperparameter_sweep():
        """
        Crée une configuration de sweep pour hyperparameter tuning
        """

        sweep_config = {
            'method': 'bayes',  # ou 'grid', 'random'
            'metric': {
                'name': 'validation/policy_categorical_accuracy',
                'goal': 'maximize'
            },
            'parameters': {
                'initial_lr': {
                    'distribution': 'log_uniform_values',
                    'min': 0.001,
                    'max': 0.1
                },
                'batch_size': {
                    'values': [16, 32, 64, 128]
                },
                'expansion_factor': {
                    'values': [2, 4, 6, 8]
                },
                'dropout_rate': {
                    'distribution': 'uniform',
                    'min': 0.1,
                    'max': 0.5
                },
                'policy_weight': {
                    'distribution': 'uniform',
                    'min': 0.5,
                    'max': 2.0
                },
                'value_weight': {
                    'distribution': 'uniform',
                    'min': 0.5,
                    'max': 2.0
                },
                'cosine_cycles': {
                    'values': [1, 2, 4, 8]
                }
            }
        }

        return sweep_config

    @staticmethod
    def run_sweep_agent(train_function):
        """
        Lance un agent de sweep
        """

        def sweep_train():
            # Initialiser wandb pour le sweep
            wandb.init()

            # Récupérer la config du sweep
            config = wandb.config

            # Appeler votre fonction d'entraînement avec les hyperparamètres
            train_function(
                batch=config.batch_size,
                initial_lr=config.initial_lr,
                policy_weight=config.policy_weight,
                value_weight=config.value_weight,
                nb_cosinedecay_cycle=config.cosine_cycles,
                wandb_config=dict(config)
            )

        return sweep_train

    @staticmethod
    def compare_experiments(run_ids: List[str], metrics: List[str]):
        """
        Compare plusieurs expériences
        """
        api = wandb.Api()

        comparison_data = []

        for run_id in run_ids:
            run = api.run(f"your_entity/go-mobilenet-research/{run_id}")

            run_data = {
                'run_id': run_id,
                'name': run.name,
                'state': run.state,
                'config': run.config
            }

            # Récupérer les métriques finales
            for metric in metrics:
                run_data[metric] = run.summary.get(metric, None)

            comparison_data.append(run_data)

        return pd.DataFrame(comparison_data)


# ============================================================================
# 4. SYSTÈME D'ALERTES
# ============================================================================

class WandbAlertSystem:
    """Système d'alertes intelligent"""

    def __init__(self, slack_webhook=None, email=None):
        self.slack_webhook = slack_webhook
        self.email = email

    def check_and_alert(self, metrics: Dict):
        """Vérifie les métriques et envoie des alertes si nécessaire"""

        alerts = []

        # Alertes de performance
        if metrics.get('validation/policy_categorical_accuracy', 0) < 0.3:
            alerts.append({
                'level': 'WARNING',
                'message': 'Policy accuracy très faible (<30%)',
                'metric': 'policy_accuracy',
                'value': metrics['validation/policy_categorical_accuracy']
            })

        if metrics.get('validation/value_mse', float('inf')) > 0.5:
            alerts.append({
                'level': 'CRITICAL',
                'message': 'Value MSE très élevée (>0.5)',
                'metric': 'value_mse',
                'value': metrics['validation/value_mse']
            })

        # Envoyer les alertes
        for alert in alerts:
            self._send_alert(alert)

            # Log aussi dans WandB
            wandb.log({
                f"alerts/{alert['level'].lower()}": 1,
                f"alerts/{alert['metric']}_issue": alert['value']
            })

    def _send_alert(self, alert: Dict):
        """Envoie une alerte via Slack/Email"""
        message = f"🚨 {alert['level']}: {alert['message']} (Value: {alert['value']:.4f})"

        if self.slack_webhook:
            # Envoyer vers Slack
            import requests
            requests.post(self.slack_webhook, json={'text': message})

        print(f"⚠️  ALERT: {message}")


# ============================================================================
# 5. UTILITAIRES ET FONCTIONS D'AIDE
# ============================================================================

def save_wandb_config(config: Dict, filename: str = "wandb_config.json"):
    """Sauvegarde la configuration WandB"""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📄 Configuration WandB sauvegardée: {filename}")


def load_wandb_config(filename: str = "wandb_config.json") -> Dict:
    """Charge la configuration WandB"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Fichier {filename} non trouvé, utilisation config par défaut")
        return {}


def create_experiment_summary(run_url: str, metrics: Dict, save_path: str = "experiment_summary.md"):
    """Crée un résumé d'expérience en Markdown"""

    summary = f"""# Expérience Go MobileNet

## 🔗 Liens
- **WandB Run**: {run_url}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Métriques Finales
- **Policy Accuracy**: {metrics.get('final_policy_accuracy', 'N/A'):.4f}
- **Value MSE**: {metrics.get('final_value_mse', 'N/A'):.4f}
- **Training Time**: {metrics.get('total_training_time', 'N/A'):.2f}s
- **Best Validation Loss**: {metrics.get('best_validation_loss', 'N/A'):.4f}

## 🎯 Résultats
{metrics.get('experiment_notes', 'Aucune note spécifique')}

---
*Généré automatiquement par wandb_integration.py*
"""

    with open(save_path, 'w') as f:
        f.write(summary)

    print(f"📋 Résumé d'expérience sauvegardé: {save_path}")


# ============================================================================
# 6. EXEMPLE D'UTILISATION
# ============================================================================

def example_usage():
    """
    Exemple d'utilisation du système WandB
    """

    print("🚀 Exemple d'utilisation WandB Integration")
    print("=" * 50)

    # 1. Créer le tracker
    tracker = WandbGoTracker(
        project_name="go-mobilenet-demo",
        experiment_name="demo_integration"
    )

    # 2. Configuration d'expérience
    config = {
        "model_type": "MobileNet_Demo",
        "epochs": 10,
        "batch_size": 32,
        "notes": "Démonstration de l'intégration WandB"
    }

    # 3. Initialiser WandB
    run = tracker.initialize_run(
        config=config,
        tags=["demo", "integration"],
        notes="Test de l'intégration WandB avec le projet Go"
    )

    print(f"🌐 Run WandB: {run.url}")

    # 4. Simuler quelques métriques
    for epoch in range(1, 11):
        metrics = {
            "train/loss": 2.0 - 0.1 * epoch + np.random.normal(0, 0.05),
            "train/policy_categorical_accuracy": 0.3 + 0.02 * epoch + np.random.normal(0, 0.01),
            "train/value_mse": 0.25 - 0.01 * epoch + np.random.normal(0, 0.005),
            "optimization/learning_rate": 0.02 * (1 + np.cos(np.pi * epoch / 10)) / 2
        }

        if epoch % 5 == 0:
            metrics.update({
                "validation/policy_categorical_accuracy": metrics["train/policy_categorical_accuracy"] - 0.02,
                "validation/value_mse": metrics["train/value_mse"] + 0.01
            })

        tracker.log_metrics(metrics, step=epoch)
        print(f"✅ Epoch {epoch}: Métriques logged")

    # 5. Terminer
    tracker.finish_run()

    print("✅ Démonstration terminée!")
    print("\n📊 Consultez vos résultats sur: https://wandb.ai/")


if __name__ == "__main__":
    print("📦 WandB Integration pour Go MobileNet")
    print("🎯 Classes principales:")
    print("   • WandbGoTracker: Tracker principal")
    print("   • WandbGoCallbackManual: Callback pour appel manuel")
    print("   • WandbExperimentManager: Gestion des sweeps")
    print("   • WandbAlertSystem: Système d'alertes")

    try:
        example_usage()
    except Exception as e:
        print(f"❌ Erreur: {e}")
        print("\n💡 Assurez-vous d'avoir:")
        print("   1. Installé wandb: pip install wandb")
        print("   2. Configuré votre compte: wandb login")
        print("   3. Une connexion internet active")