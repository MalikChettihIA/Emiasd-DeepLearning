# ============================================================================
# SYSTÈME DE DIAGNOSTIC DE CONVERGENCE POUR GO MOBILENET
# Fichier: convergence_diagnostics.py
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import logging
from typing import Dict, List, Tuple, Optional
import warnings
from tensorflow.keras.callbacks import Callback


class ConvergenceDiagnostics:
    """
    Système de diagnostic de convergence pour les réseaux de neurones Go
    Basé sur votre projet go_train.py et les métriques académiques
    """

    def __init__(self, log_level=logging.INFO):
        self.setup_logging(log_level)
        self.metrics_history = defaultdict(list)
        self.alerts = []
        self.benchmarks = self._load_benchmarks()

    def setup_logging(self, level):
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def _load_benchmarks(self) -> Dict:
        """Benchmarks basés sur la recherche académique"""
        return {
            'policy_accuracy': {
                'mobilenet_small': 0.55,  # 16 blocs MobileNet
                'mobilenet_large': 0.61,  # 48+ blocs MobileNet
                'resnet_baseline': 0.475,  # ResNet 20 blocs
                'alpha_go': 0.57,  # AlphaGo paper
            },
            'value_mse': {
                'good': 0.15,  # Bon modèle
                'average': 0.20,  # Modèle moyen
                'poor': 0.25,  # Modèle faible
            },
            'learning_rate': {
                'initial': 0.02,  # Votre config actuelle
                'min_effective': 1e-6,  # LR minimum utile
                'cosine_final': 0.0,  # Cosine annealing
            }
        }

    def record_epoch(self, epoch: int, metrics: Dict, validation_metrics: Optional[Dict] = None):
        """Enregistre les métriques d'une époque"""

        # Métriques d'entraînement
        for key, value in metrics.items():
            self.metrics_history[key].append({
                'epoch': epoch,
                'value': value,
                'type': 'train'
            })

        # Métriques de validation
        if validation_metrics:
            for key, value in validation_metrics.items():
                val_key = f"val_{key}" if not key.startswith('val_') else key
                self.metrics_history[val_key].append({
                    'epoch': epoch,
                    'value': value,
                    'type': 'validation'
                })

    def analyze_convergence(self) -> Dict:
        """Analyse complète de la convergence"""

        results = {
            'overall_health': 'UNKNOWN',
            'issues': [],
            'recommendations': [],
            'metrics_analysis': {}
        }

        # 1. Analyse des métriques principales
        results['metrics_analysis']['policy'] = self._analyze_policy_convergence()
        results['metrics_analysis']['value'] = self._analyze_value_convergence()
        results['metrics_analysis']['learning_rate'] = self._analyze_learning_rate()
        results['metrics_analysis']['stability'] = self._analyze_training_stability()

        # 2. Détection des problèmes
        issues = []
        issues.extend(self._detect_overfitting())
        issues.extend(self._detect_underfitting())
        issues.extend(self._detect_gradient_problems())
        issues.extend(self._detect_architecture_issues())

        results['issues'] = issues

        # 3. Recommandations
        results['recommendations'] = self._generate_recommendations(issues)

        # 4. Score de santé global
        results['overall_health'] = self._compute_health_score(issues)

        return results

    def _analyze_policy_convergence(self) -> Dict:
        """Analyse spécifique de la convergence de la policy"""

        if 'policy_categorical_accuracy' not in self.metrics_history:
            return {'status': 'NO_DATA'}

        accuracies = [m['value'] for m in self.metrics_history['policy_categorical_accuracy']]
        epochs = [m['epoch'] for m in self.metrics_history['policy_categorical_accuracy']]

        analysis = {
            'current_accuracy': accuracies[-1] if accuracies else 0,
            'max_accuracy': max(accuracies) if accuracies else 0,
            'improvement_rate': self._calculate_improvement_rate(accuracies),
            'plateau_detected': self._detect_plateau(accuracies, threshold=0.001, window=10),
            'benchmark_comparison': self._compare_to_benchmarks(accuracies[-1] if accuracies else 0, 'policy_accuracy')
        }

        return analysis

    def _analyze_value_convergence(self) -> Dict:
        """Analyse spécifique de la convergence de la value"""

        if 'value_mse' not in self.metrics_history:
            return {'status': 'NO_DATA'}

        mse_values = [m['value'] for m in self.metrics_history['value_mse']]

        analysis = {
            'current_mse': mse_values[-1] if mse_values else float('inf'),
            'min_mse': min(mse_values) if mse_values else float('inf'),
            'convergence_rate': self._calculate_convergence_rate(mse_values),
            'stagnation_detected': self._detect_stagnation(mse_values),
            'benchmark_comparison': self._compare_to_benchmarks(mse_values[-1] if mse_values else float('inf'),
                                                                'value_mse')
        }

        return analysis

    def _analyze_learning_rate(self) -> Dict:
        """Analyse du learning rate"""

        if 'learning_rate' not in self.metrics_history:
            return {'status': 'NO_DATA'}

        lrs = [m['value'] for m in self.metrics_history['learning_rate']]

        analysis = {
            'current_lr': lrs[-1] if lrs else 0,
            'lr_schedule_health': self._evaluate_lr_schedule(lrs),
            'cosine_annealing_detected': self._detect_cosine_schedule(lrs),
            'effective_range': len([lr for lr in lrs if lr > self.benchmarks['learning_rate']['min_effective']])
        }

        return analysis

    def _analyze_training_stability(self) -> Dict:
        """Analyse de la stabilité d'entraînement"""

        loss_values = []
        if 'loss' in self.metrics_history:
            loss_values = [m['value'] for m in self.metrics_history['loss']]

        analysis = {
            'loss_variance': np.var(loss_values[-20:]) if len(loss_values) >= 20 else float('inf'),
            'oscillations_detected': self._detect_oscillations(loss_values),
            'divergence_detected': self._detect_divergence(loss_values),
            'nan_detected': any(np.isnan(v) or np.isinf(v) for v in loss_values)
        }

        return analysis

    def _detect_overfitting(self) -> List[Dict]:
        """Détecte l'overfitting"""
        issues = []

        # Comparaison train vs validation
        if ('policy_categorical_accuracy' in self.metrics_history and
                'val_policy_categorical_accuracy' in self.metrics_history):

            train_acc = [m['value'] for m in self.metrics_history['policy_categorical_accuracy']]
            val_acc = [m['value'] for m in self.metrics_history['val_policy_categorical_accuracy']]

            if len(train_acc) >= 10 and len(val_acc) >= 10:
                recent_gap = np.mean(train_acc[-10:]) - np.mean(val_acc[-10:])

                if recent_gap > 0.05:  # 5% d'écart
                    issues.append({
                        'type': 'OVERFITTING',
                        'severity': 'HIGH' if recent_gap > 0.10 else 'MEDIUM',
                        'description': f'Gap train-val de {recent_gap:.3f} détecté',
                        'metric': 'policy_accuracy'
                    })

        return issues

    def _detect_underfitting(self) -> List[Dict]:
        """Détecte l'underfitting"""
        issues = []

        if 'policy_categorical_accuracy' in self.metrics_history:
            accuracies = [m['value'] for m in self.metrics_history['policy_categorical_accuracy']]

            if len(accuracies) >= 50:  # Après 50 époques
                current_acc = accuracies[-1]
                expected_min = self.benchmarks['policy_accuracy']['mobilenet_small']

                if current_acc < expected_min * 0.9:  # 90% du benchmark minimum
                    issues.append({
                        'type': 'UNDERFITTING',
                        'severity': 'HIGH',
                        'description': f'Accuracy {current_acc:.3f} sous le benchmark {expected_min:.3f}',
                        'metric': 'policy_accuracy'
                    })

        return issues

    def _detect_gradient_problems(self) -> List[Dict]:
        """Détecte les problèmes de gradients"""
        issues = []

        if 'loss' in self.metrics_history:
            losses = [m['value'] for m in self.metrics_history['loss']]

            # Explosion des gradients
            if len(losses) >= 5:
                recent_losses = losses[-5:]
                if any(loss > losses[0] * 10 for loss in recent_losses):
                    issues.append({
                        'type': 'GRADIENT_EXPLOSION',
                        'severity': 'CRITICAL',
                        'description': 'Loss a explosé récemment',
                        'metric': 'loss'
                    })

            # Gradients qui disparaissent
            if len(losses) >= 20:
                improvement = losses[-20] - losses[-1]
                if improvement < 0.001:  # Très peu d'amélioration
                    issues.append({
                        'type': 'GRADIENT_VANISHING',
                        'severity': 'MEDIUM',
                        'description': 'Très peu d\'amélioration sur 20 époques',
                        'metric': 'loss'
                    })

        return issues

    def _detect_architecture_issues(self) -> List[Dict]:
        """Détecte les problèmes d'architecture (comme le bug bottleneck)"""
        issues = []

        # Problème potentiel de bottleneck si convergence très lente
        if ('policy_categorical_accuracy' in self.metrics_history and
                len(self.metrics_history['policy_categorical_accuracy']) >= 100):

            accuracies = [m['value'] for m in self.metrics_history['policy_categorical_accuracy']]

            # Si après 100 époques, on n'atteint pas 45% (très bas)
            if accuracies[-1] < 0.45:
                issues.append({
                    'type': 'ARCHITECTURE_BUG',
                    'severity': 'HIGH',
                    'description': 'Convergence anormalement lente - vérifier bottleneck_block',
                    'metric': 'policy_accuracy',
                    'suggestion': 'Vérifier _bottleneck_block expansion calculation'
                })

        return issues

    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Génère des recommandations basées sur les problèmes détectés"""
        recommendations = []

        issue_types = [issue['type'] for issue in issues]

        if 'OVERFITTING' in issue_types:
            recommendations.extend([
                "🔧 Augmenter le dropout rate (actuellement 0.2)",
                "🔧 Augmenter la régularisation L2",
                "🔧 Réduire la complexité du modèle (moins de blocs)",
                "🔧 Implémenter early stopping"
            ])

        if 'UNDERFITTING' in issue_types:
            recommendations.extend([
                "🔧 Augmenter la capacité du modèle (plus de filtres/blocs)",
                "🔧 Réduire la régularisation",
                "🔧 Vérifier la qualité des données d'entraînement",
                "🔧 Augmenter le learning rate initial"
            ])

        if 'GRADIENT_EXPLOSION' in issue_types:
            recommendations.extend([
                "🚨 URGENT: Réduire drastiquement le learning rate",
                "🔧 Implémenter gradient clipping (clipnorm=1.0)",
                "🔧 Vérifier la normalisation des données"
            ])

        if 'GRADIENT_VANISHING' in issue_types:
            recommendations.extend([
                "🔧 Vérifier les connexions résiduelles",
                "🔧 Utiliser l'activation Swish au lieu de ReLU",
                "🔧 Réduire la profondeur du réseau"
            ])

        if 'ARCHITECTURE_BUG' in issue_types:
            recommendations.extend([
                "🚨 CRITIQUE: Corriger le bug dans _bottleneck_block",
                "🔧 Vérifier: expansion = int(input_filters * factor)",
                "🔧 Tester avec un ResNet simple pour comparaison"
            ])

        return recommendations

    def plot_diagnostics(self, save_path: Optional[str] = None):
        """Génère les graphiques de diagnostic"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔍 Diagnostic de Convergence - Projet Go MobileNet', fontsize=16)

        # 1. Policy Accuracy
        if 'policy_categorical_accuracy' in self.metrics_history:
            train_data = self.metrics_history['policy_categorical_accuracy']
            epochs = [d['epoch'] for d in train_data]
            values = [d['value'] for d in train_data]

            axes[0, 0].plot(epochs, values, 'b-', label='Train', linewidth=2)

            # Validation si disponible
            if 'val_policy_categorical_accuracy' in self.metrics_history:
                val_data = self.metrics_history['val_policy_categorical_accuracy']
                val_epochs = [d['epoch'] for d in val_data]
                val_values = [d['value'] for d in val_data]
                axes[0, 0].plot(val_epochs, val_values, 'r--', label='Validation', linewidth=2)

            # Benchmarks
            axes[0, 0].axhline(y=self.benchmarks['policy_accuracy']['mobilenet_small'],
                               color='green', linestyle=':', label='MobileNet Benchmark')
            axes[0, 0].axhline(y=self.benchmarks['policy_accuracy']['resnet_baseline'],
                               color='orange', linestyle=':', label='ResNet Baseline')

            axes[0, 0].set_title('Policy Accuracy Evolution')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # 2. Value MSE
        if 'value_mse' in self.metrics_history:
            train_data = self.metrics_history['value_mse']
            epochs = [d['epoch'] for d in train_data]
            values = [d['value'] for d in train_data]

            axes[0, 1].plot(epochs, values, 'b-', label='Train MSE', linewidth=2)
            axes[0, 1].axhline(y=self.benchmarks['value_mse']['good'],
                               color='green', linestyle=':', label='Good MSE')
            axes[0, 1].axhline(y=self.benchmarks['value_mse']['poor'],
                               color='red', linestyle=':', label='Poor MSE')

            axes[0, 1].set_title('Value MSE Evolution')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_yscale('log')

        # 3. Learning Rate
        if 'learning_rate' in self.metrics_history:
            lr_data = self.metrics_history['learning_rate']
            epochs = [d['epoch'] for d in lr_data]
            values = [d['value'] for d in lr_data]

            axes[0, 2].plot(epochs, values, 'purple', linewidth=2)
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Learning Rate')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].set_yscale('log')

        # 4. Loss Evolution
        if 'loss' in self.metrics_history:
            loss_data = self.metrics_history['loss']
            epochs = [d['epoch'] for d in loss_data]
            values = [d['value'] for d in loss_data]

            axes[1, 0].plot(epochs, values, 'red', linewidth=2)
            axes[1, 0].set_title('Total Loss Evolution')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # 5. Policy vs Value Loss
        if 'policy_loss' in self.metrics_history and 'value_loss' in self.metrics_history:
            policy_data = self.metrics_history['policy_loss']
            value_data = self.metrics_history['value_loss']

            p_epochs = [d['epoch'] for d in policy_data]
            p_values = [d['value'] for d in policy_data]
            v_epochs = [d['epoch'] for d in value_data]
            v_values = [d['value'] for d in value_data]

            axes[1, 1].plot(p_epochs, p_values, 'blue', label='Policy Loss', linewidth=2)
            axes[1, 1].plot(v_epochs, v_values, 'orange', label='Value Loss', linewidth=2)
            axes[1, 1].set_title('Policy vs Value Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Training Stability (Loss Variance)
        if 'loss' in self.metrics_history:
            loss_data = self.metrics_history['loss']
            values = [d['value'] for d in loss_data]

            # Rolling variance
            window = 10
            rolling_var = []
            epochs_var = []

            for i in range(window, len(values)):
                variance = np.var(values[i - window:i])
                rolling_var.append(variance)
                epochs_var.append(loss_data[i]['epoch'])

            axes[1, 2].plot(epochs_var, rolling_var, 'green', linewidth=2)
            axes[1, 2].set_title(f'Training Stability (Rolling Variance, window={window})')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss Variance')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Diagnostic plots saved to {save_path}")

        plt.show()

    def print_summary(self, analysis_results: Dict):
        """Affiche un résumé textuel du diagnostic"""

        print("🏥 " + "=" * 80)
        print("🔍 RAPPORT DE DIAGNOSTIC DE CONVERGENCE")
        print("=" * 84)
        print()

        # Santé globale
        health = analysis_results['overall_health']
        health_emoji = {'HEALTHY': '💚', 'WARNING': '🟡', 'CRITICAL': '🔴', 'UNKNOWN': '❓'}
        print(f"📊 SANTÉ GLOBALE: {health_emoji.get(health, '❓')} {health}")
        print()

        # Métriques actuelles
        print("📈 MÉTRIQUES ACTUELLES:")
        print("-" * 40)

        if 'policy' in analysis_results['metrics_analysis']:
            policy_analysis = analysis_results['metrics_analysis']['policy']
            if 'current_accuracy' in policy_analysis:
                acc = policy_analysis['current_accuracy']
                print(f"   Policy Accuracy: {acc:.4f}")

                # Comparaison benchmarks
                if acc > self.benchmarks['policy_accuracy']['mobilenet_large']:
                    print("     ✅ Excellent (> MobileNet Large)")
                elif acc > self.benchmarks['policy_accuracy']['mobilenet_small']:
                    print("     ✅ Bon (> MobileNet Small)")
                elif acc > self.benchmarks['policy_accuracy']['resnet_baseline']:
                    print("     ⚠️  Acceptable (> ResNet baseline)")
                else:
                    print("     ❌ Sous-performance")

        if 'value' in analysis_results['metrics_analysis']:
            value_analysis = analysis_results['metrics_analysis']['value']
            if 'current_mse' in value_analysis:
                mse = value_analysis['current_mse']
                print(f"   Value MSE: {mse:.4f}")

                if mse < self.benchmarks['value_mse']['good']:
                    print("     ✅ Excellent")
                elif mse < self.benchmarks['value_mse']['average']:
                    print("     ✅ Bon")
                elif mse < self.benchmarks['value_mse']['poor']:
                    print("     ⚠️  Acceptable")
                else:
                    print("     ❌ Problématique")

        print()

        # Problèmes détectés
        issues = analysis_results['issues']
        if issues:
            print("🚨 PROBLÈMES DÉTECTÉS:")
            print("-" * 40)

            for issue in issues:
                severity_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                emoji = severity_emoji.get(issue['severity'], '❓')
                print(f"   {emoji} {issue['type']}: {issue['description']}")
            print()

        # Recommandations
        recommendations = analysis_results['recommendations']
        if recommendations:
            print("💡 RECOMMANDATIONS:")
            print("-" * 40)
            for rec in recommendations:
                print(f"   {rec}")
            print()

        print("=" * 84)

    # Méthodes utilitaires
    def _calculate_improvement_rate(self, values: List[float], window: int = 10) -> float:
        if len(values) < window:
            return 0.0
        return (values[-1] - values[-window]) / window

    def _calculate_convergence_rate(self, values: List[float], window: int = 10) -> float:
        if len(values) < window:
            return 0.0
        return (values[-window] - values[-1]) / window  # Pour MSE, on veut une diminution

    def _detect_plateau(self, values: List[float], threshold: float = 0.001, window: int = 10) -> bool:
        if len(values) < window:
            return False
        recent_variance = np.var(values[-window:])
        return recent_variance < threshold

    def _detect_stagnation(self, values: List[float], window: int = 20) -> bool:
        if len(values) < window:
            return False
        improvement = values[-window] - values[-1]
        return improvement < 0.001

    def _detect_oscillations(self, values: List[float], window: int = 20) -> bool:
        if len(values) < window:
            return False
        recent_values = values[-window:]
        # Détecte si la variance est trop élevée
        variance = np.var(recent_values)
        mean_val = np.mean(recent_values)
        return variance > (mean_val * 0.1) ** 2

    def _detect_divergence(self, values: List[float], window: int = 10) -> bool:
        if len(values) < window:
            return False
        recent_trend = np.polyfit(range(window), values[-window:], 1)[0]
        return recent_trend > 0.01  # Loss qui augmente

    def _evaluate_lr_schedule(self, lrs: List[float]) -> str:
        if not lrs:
            return 'NO_DATA'

        # Vérifie si le LR diminue globalement
        if lrs[-1] < lrs[0] * 0.1:
            return 'GOOD_DECAY'
        elif lrs[-1] < lrs[0] * 0.5:
            return 'MODERATE_DECAY'
        else:
            return 'INSUFFICIENT_DECAY'

    def _detect_cosine_schedule(self, lrs: List[float]) -> bool:
        if len(lrs) < 50:
            return False

        # Détecte le pattern cosinus (approximatif)
        # Le LR devrait diminuer de façon non-linéaire
        mid_point = len(lrs) // 2
        first_half_avg_change = np.mean(np.diff(lrs[:mid_point]))
        second_half_avg_change = np.mean(np.diff(lrs[mid_point:]))

        # Dans cosine annealing, la décroissance ralentit
        return abs(second_half_avg_change) < abs(first_half_avg_change) * 0.5

    def _compare_to_benchmarks(self, value: float, benchmark_type: str) -> str:
        if benchmark_type not in self.benchmarks:
            return 'NO_BENCHMARK'

        benchmarks = self.benchmarks[benchmark_type]

        if benchmark_type == 'policy_accuracy':
            if value > benchmarks['mobilenet_large']:
                return 'EXCELLENT'
            elif value > benchmarks['mobilenet_small']:
                return 'GOOD'
            elif value > benchmarks['resnet_baseline']:
                return 'ACCEPTABLE'
            else:
                return 'POOR'

        elif benchmark_type == 'value_mse':
            if value < benchmarks['good']:
                return 'EXCELLENT'
            elif value < benchmarks['average']:
                return 'GOOD'
            elif value < benchmarks['poor']:
                return 'ACCEPTABLE'
            else:
                return 'POOR'

        return 'UNKNOWN'

    def _compute_health_score(self, issues: List[Dict]) -> str:
        if not issues:
            return 'HEALTHY'

        critical_count = sum(1 for issue in issues if issue['severity'] == 'CRITICAL')
        high_count = sum(1 for issue in issues if issue['severity'] == 'HIGH')

        if critical_count > 0:
            return 'CRITICAL'
        elif high_count > 0:
            return 'WARNING'
        else:
            return 'HEALTHY'


# ============================================================================
# CALLBACK DIAGNOSTIC MANUEL
# ============================================================================

class DiagnosticCallbackManual(Callback):
    """
    Callback de diagnostic conçu pour être appelé manuellement
    Compatible avec la structure epochs=1 de go_train.py
    """

    def __init__(self,
                 diagnostics_instance: ConvergenceDiagnostics,
                 alert_frequency: int = 20,
                 save_plots_frequency: int = 100,
                 auto_save_checkpoints: bool = True):
        """
        Args:
            diagnostics_instance: Instance de ConvergenceDiagnostics
            alert_frequency: Fréquence des analyses de diagnostic (epochs)
            save_plots_frequency: Fréquence de sauvegarde des graphiques
            auto_save_checkpoints: Sauvegarder automatiquement en cas de problème critique
        """
        super().__init__()
        self.diagnostics = diagnostics_instance
        self.alert_frequency = alert_frequency
        self.save_plots_frequency = save_plots_frequency
        self.auto_save_checkpoints = auto_save_checkpoints

        # État interne
        self.last_analysis = None
        self.critical_issues_count = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Méthode appelée manuellement avec le VRAI numéro d'époque

        Args:
            epoch: Le vrai numéro d'époque (1, 2, 3, ...) - PAS celui de Keras
            logs: Dictionnaire des métriques
        """
        logs = logs or {}

        # Convertir epoch en base 1 pour cohérence
        true_epoch = epoch + 1 if epoch >= 0 else 1

        # Toujours enregistrer les métriques
        self.diagnostics.record_epoch(true_epoch, logs)

        # Analyse périodique
        if true_epoch % self.alert_frequency == 0:
            self._perform_diagnostic_analysis(true_epoch)

        # Sauvegarde des graphiques
        if true_epoch % self.save_plots_frequency == 0:
            self._save_diagnostic_plots(true_epoch)

        # Détection de problèmes critiques à chaque époque
        self._check_critical_issues(true_epoch, logs)

    def log_validation_manually(self, epoch, val_metrics):
        """
        Log manuel des métriques de validation pour le diagnostic

        Args:
            epoch: Numéro d'époque (base 1)
            val_metrics: Dict des métriques de validation
        """
        self.diagnostics.record_epoch(epoch, {}, val_metrics)
        print(f"🔍 Diagnostic: Recorded validation metrics for epoch {epoch}")

    def log_learning_rate_manually(self, epoch, lr):
        """Log manuel du learning rate pour le diagnostic"""
        self.diagnostics.record_epoch(epoch, {'learning_rate': lr})

    def force_analysis(self, epoch):
        """Force une analyse de diagnostic à une époque donnée"""
        self._perform_diagnostic_analysis(epoch)
        return self.last_analysis

    def get_current_health_status(self):
        """Retourne le statut de santé actuel"""
        if self.last_analysis:
            return self.last_analysis.get('overall_health', 'UNKNOWN')
        return 'UNKNOWN'

    def _perform_diagnostic_analysis(self, epoch):
        """Effectue l'analyse de diagnostic complète"""
        print(f"\n🔍 DIAGNOSTIC ANALYSIS - Epoch {epoch}")
        print("-" * 50)

        # Analyse complète
        self.last_analysis = self.diagnostics.analyze_convergence()

        # Affichage du résumé
        self.diagnostics.print_summary(self.last_analysis)

        # Actions automatiques basées sur les résultats
        self._handle_analysis_results(epoch, self.last_analysis)

        return self.last_analysis

    def _save_diagnostic_plots(self, epoch):
        """Sauvegarde les graphiques de diagnostic"""
        filename = f'diagnostics_epoch_{epoch}.png'
        try:
            self.diagnostics.plot_diagnostics(filename)
            print(f"📊 Diagnostic plots saved: {filename}")
        except Exception as e:
            print(f"⚠️  Failed to save diagnostic plots: {e}")

    def _check_critical_issues(self, epoch, logs):
        """Vérifie les problèmes critiques à chaque époque"""

        # Détection de NaN/Inf
        for key, value in logs.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                print(f"🚨 CRITICAL: NaN/Inf detected in {key} at epoch {epoch}!")
                self._handle_critical_issue(epoch, f"NaN_in_{key}")
                return

        # Détection d'explosion de loss
        current_loss = logs.get('loss', 0)
        if hasattr(self, '_baseline_loss'):
            if current_loss > self._baseline_loss * 5:  # 5x explosion
                print(f"🚨 CRITICAL: Loss explosion detected at epoch {epoch}!")
                print(f"   Current: {current_loss:.4f}, Baseline: {self._baseline_loss:.4f}")
                self._handle_critical_issue(epoch, "loss_explosion")
        else:
            # Établir baseline
            if epoch <= 5:  # Premières époques
                self._baseline_loss = current_loss

    def _handle_analysis_results(self, epoch, analysis):
        """Gère les résultats de l'analyse automatiquement"""

        health = analysis.get('overall_health', 'UNKNOWN')
        issues = analysis.get('issues', [])

        # Compter les problèmes critiques
        critical_issues = [i for i in issues if i.get('severity') == 'CRITICAL']

        if critical_issues:
            self.critical_issues_count += len(critical_issues)
            print(f"\n🚨 {len(critical_issues)} CRITICAL ISSUE(S) DETECTED:")
            for issue in critical_issues:
                print(f"   • {issue['type']}: {issue['description']}")

            # Sauvegarder automatiquement en cas de problème critique
            if self.auto_save_checkpoints and hasattr(self, 'model'):
                self._emergency_checkpoint(epoch)

        # Recommandations automatiques
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            print(f"\n💡 AUTOMATED RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Top 3 recommandations
                print(f"   {rec}")

    def _handle_critical_issue(self, epoch, issue_type):
        """Gère un problème critique détecté"""
        self.critical_issues_count += 1

        print(f"\n🚨 CRITICAL ISSUE HANDLER ACTIVATED")
        print(f"   Issue: {issue_type}")
        print(f"   Epoch: {epoch}")
        print(f"   Total critical issues: {self.critical_issues_count}")

        # Sauvegarder automatiquement
        if self.auto_save_checkpoints and hasattr(self, 'model'):
            self._emergency_checkpoint(epoch, issue_type)

        # Suggérer des actions
        if issue_type.startswith("NaN"):
            print("   🔧 SUGGESTED ACTION: Reduce learning rate immediately")
            print("   🔧 SUGGESTED ACTION: Check data preprocessing")
        elif issue_type == "loss_explosion":
            print("   🔧 SUGGESTED ACTION: Reduce learning rate by 10x")
            print("   🔧 SUGGESTED ACTION: Enable gradient clipping")

        return self.critical_issues_count >= 3  # Arrêter si 3+ problèmes critiques

    def _emergency_checkpoint(self, epoch, reason="critical_issue"):
        """Sauvegarde d'urgence du modèle"""
        if hasattr(self, 'model'):
            filename = f"emergency_model_epoch_{epoch}_{reason}.h5"
            try:
                self.model.save(filename)
                print(f"💾 Emergency checkpoint saved: {filename}")
            except Exception as e:
                print(f"❌ Failed to save emergency checkpoint: {e}")

    def set_model(self, model):
        """Définit le modèle pour les sauvegardes d'urgence"""
        self.model = model


# ============================================================================
# ANALYSES SPÉCIALISÉES POUR LE GO
# ============================================================================

class GoSpecificAnalysis:
    """Analyses spécifiques au domaine du Go"""

    @staticmethod
    def analyze_policy_distribution(policy_outputs: np.ndarray) -> Dict:
        """Analyse la distribution des prédictions de policy"""

        # Entropie de la distribution
        entropy = -np.sum(policy_outputs * np.log(policy_outputs + 1e-8), axis=1)

        # Sparsité (concentration sur few moves)
        top1_prob = np.max(policy_outputs, axis=1)
        top3_prob = np.sum(np.sort(policy_outputs, axis=1)[:, -3:], axis=1)

        analysis = {
            'mean_entropy': np.mean(entropy),
            'entropy_std': np.std(entropy),
            'mean_top1_confidence': np.mean(top1_prob),
            'mean_top3_confidence': np.mean(top3_prob),
            'distribution_health': 'GOOD' if 2.0 < np.mean(entropy) < 4.0 else 'POOR'
        }

        return analysis

    @staticmethod
    def analyze_value_predictions(value_outputs: np.ndarray, true_values: np.ndarray) -> Dict:
        """Analyse les prédictions de value"""

        # Corrélation avec les vraies valeurs
        correlation = np.corrcoef(value_outputs.flatten(), true_values.flatten())[0, 1]

        # Distribution des prédictions
        pred_mean = np.mean(value_outputs)
        pred_std = np.std(value_outputs)

        # Calibration (prédictions proches de 0.5 sont-elles vraiment incertaines?)
        uncertain_mask = (value_outputs > 0.4) & (value_outputs < 0.6)
        uncertain_accuracy = np.mean(np.abs(value_outputs[uncertain_mask] - true_values[uncertain_mask]))

        analysis = {
            'correlation': correlation,
            'prediction_mean': pred_mean,
            'prediction_std': pred_std,
            'uncertain_accuracy': uncertain_accuracy,
            'calibration_health': 'GOOD' if correlation > 0.7 else 'POOR'
        }

        return analysis


# ============================================================================
# SYSTÈME D'ALERTES EN TEMPS RÉEL
# ============================================================================

class RealTimeMonitor:
    """Monitoring en temps réel pendant l'entraînement"""

    def __init__(self, alert_thresholds: Optional[Dict] = None):
        self.thresholds = alert_thresholds or {
            'loss_explosion': 5.0,  # Si loss > 5x la valeur initiale
            'accuracy_drop': 0.05,  # Chute de 5% d'accuracy
            'gradient_norm': 10.0,  # Gradient norm trop élevé
            'nan_detection': True,  # Détection de NaN
        }
        self.baseline_loss = None
        self.last_accuracy = None
        self.alert_history = []

    def check_batch(self, batch_idx: int, metrics: Dict) -> List[Dict]:
        """Vérifie chaque batch pour des problèmes immédiats"""
        alerts = []

        current_loss = metrics.get('loss', 0)
        current_accuracy = metrics.get('accuracy', 0)

        # Initialisation baseline
        if self.baseline_loss is None:
            self.baseline_loss = current_loss
            self.last_accuracy = current_accuracy
            return alerts

        # Explosion de loss
        if current_loss > self.baseline_loss * self.thresholds['loss_explosion']:
            alerts.append({
                'type': 'LOSS_EXPLOSION',
                'batch': batch_idx,
                'current': current_loss,
                'baseline': self.baseline_loss,
                'action': 'REDUCE_LR_IMMEDIATELY'
            })

        # Chute d'accuracy
        if (self.last_accuracy and
                current_accuracy < self.last_accuracy - self.thresholds['accuracy_drop']):
            alerts.append({
                'type': 'ACCURACY_DROP',
                'batch': batch_idx,
                'drop': self.last_accuracy - current_accuracy,
                'action': 'CHECK_DATA_CORRUPTION'
            })

        # Détection de NaN
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                alerts.append({
                    'type': 'NAN_DETECTED',
                    'batch': batch_idx,
                    'metric': key,
                    'action': 'STOP_TRAINING_IMMEDIATELY'
                })

        self.last_accuracy = current_accuracy
        self.alert_history.extend(alerts)

        return alerts


# ============================================================================
# UTILITAIRES ET FONCTIONS D'AIDE
# ============================================================================

def create_diagnostic_callback(diagnostics: ConvergenceDiagnostics):
    """Crée un callback Keras pour l'intégration automatique"""

    class DiagnosticCallback(Callback):
        def __init__(self, diagnostics_instance):
            super().__init__()
            self.diagnostics = diagnostics_instance
            self.epoch_count = 0

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_count += 1
            logs = logs or {}

            # Enregistrer toutes les métriques disponibles
            self.diagnostics.record_epoch(self.epoch_count, logs)

            # Diagnostic périodique
            if self.epoch_count % 20 == 0:
                analysis = self.diagnostics.analyze_convergence()

                if analysis['overall_health'] in ['CRITICAL', 'WARNING']:
                    print(f"\n⚠️  Diagnostic Epoch {self.epoch_count}:")
                    self.diagnostics.print_summary(analysis)

                    # Sauvegarder les graphiques
                    self.diagnostics.plot_diagnostics(f'checkpoint_epoch_{self.epoch_count}.png')

        def on_batch_end(self, batch, logs=None):
            # Monitoring en temps réel léger
            logs = logs or {}
            if any(np.isnan(v) for v in logs.values() if isinstance(v, (int, float))):
                print(f"🚨 NaN détecté au batch {batch}!")
                print(f"   Métriques: {logs}")

    return DiagnosticCallback(diagnostics)


def save_diagnostic_report(diagnostics: ConvergenceDiagnostics,
                           analysis: Dict,
                           filename: str = 'diagnostic_report.md'):
    """Sauvegarde un rapport de diagnostic en Markdown"""

    import datetime

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# 🔍 Rapport de Diagnostic de Convergence\n\n")
        f.write(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Projet**: Go MobileNet Training\n\n")

        # Santé globale
        health = analysis['overall_health']
        health_emoji = {'HEALTHY': '💚', 'WARNING': '🟡', 'CRITICAL': '🔴', 'UNKNOWN': '❓'}
        f.write(f"## 📊 Santé Globale: {health_emoji.get(health, '❓')} {health}\n\n")

        # Métriques
        f.write("## 📈 Métriques Principales\n\n")

        metrics = analysis.get('metrics_analysis', {})
        if 'policy' in metrics:
            policy = metrics['policy']
            f.write("### 🎯 Policy Network\n")
            f.write(f"- **Accuracy actuelle**: {policy.get('current_accuracy', 'N/A'):.4f}\n")
            f.write(f"- **Accuracy maximale**: {policy.get('max_accuracy', 'N/A'):.4f}\n")
            f.write(f"- **Benchmark**: {policy.get('benchmark_comparison', 'N/A')}\n")
            f.write(f"- **Plateau détecté**: {'✅' if policy.get('plateau_detected', False) else '❌'}\n\n")

        if 'value' in metrics:
            value = metrics['value']
            f.write("### 🎯 Value Network\n")
            f.write(f"- **MSE actuelle**: {value.get('current_mse', 'N/A'):.4f}\n")
            f.write(f"- **MSE minimale**: {value.get('min_mse', 'N/A'):.4f}\n")
            f.write(f"- **Benchmark**: {value.get('benchmark_comparison', 'N/A')}\n")
            f.write(f"- **Stagnation**: {'⚠️' if value.get('stagnation_detected', False) else '✅'}\n\n")

        # Problèmes
        issues = analysis.get('issues', [])
        if issues:
            f.write("## 🚨 Problèmes Détectés\n\n")
            for issue in issues:
                severity_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                emoji = severity_emoji.get(issue['severity'], '❓')
                f.write(f"- {emoji} **{issue['type']}**: {issue['description']}\n")
            f.write("\n")

        # Recommandations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            f.write("## 💡 Recommandations\n\n")
            for rec in recommendations:
                f.write(f"- {rec}\n")
            f.write("\n")

        # Footer
        f.write("---\n")
        f.write("*Rapport généré automatiquement par le système de diagnostic*\n")

    print(f"📄 Rapport sauvegardé: {filename}")


# ============================================================================
# EXEMPLE D'UTILISATION COMPLÈTE
# ============================================================================

def example_usage():
    """Exemple complet d'utilisation du système de diagnostic"""

    print("🚀 Initialisation du système de diagnostic...")

    # 1. Créer le diagnosticateur
    diagnostics = ConvergenceDiagnostics()
    monitor = RealTimeMonitor()

    # 2. Simuler des données d'entraînement (remplacez par vos vraies données)
    np.random.seed(42)

    # Simulation d'un entraînement avec amélioration progressive
    for epoch in range(1, 101):

        # Simulation de métriques d'entraînement
        base_accuracy = 0.3 + 0.003 * epoch + np.random.normal(0, 0.01)
        base_accuracy = min(base_accuracy, 0.65)  # Cap réaliste

        policy_loss = 2.0 - 0.01 * epoch + np.random.normal(0, 0.05)
        policy_loss = max(policy_loss, 0.5)

        value_mse = 0.25 - 0.001 * epoch + np.random.normal(0, 0.005)
        value_mse = max(value_mse, 0.12)

        total_loss = policy_loss + value_mse

        # Learning rate cosinus
        lr = 0.02 * (1 + np.cos(np.pi * epoch / 100)) / 2

        # Enregistrer les métriques
        train_metrics = {
            'loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_mse,
            'policy_categorical_accuracy': base_accuracy,
            'value_mse': value_mse,
            'learning_rate': lr
        }

        diagnostics.record_epoch(epoch, train_metrics)

        # Simulation validation (tous les 20 epochs)
        if epoch % 20 == 0:
            val_metrics = {
                'val_policy_loss': policy_loss + 0.1,
                'val_value_loss': value_mse + 0.02,
                'val_policy_categorical_accuracy': base_accuracy - 0.02,
                'val_value_mse': value_mse + 0.02
            }

            diagnostics.record_epoch(epoch, {}, val_metrics)

            # Diagnostic périodique
            print(f"\n🔍 Analyse à l'époque {epoch}:")
            analysis = diagnostics.analyze_convergence()
            diagnostics.print_summary(analysis)

        # Monitoring en temps réel (simulation de quelques batches par époque)
        if epoch % 10 == 0:
            batch_alerts = monitor.check_batch(epoch * 100, train_metrics)
            if batch_alerts:
                print(f"\n⚠️  Alertes batch {epoch * 100}:")
                for alert in batch_alerts:
                    print(f"   🚨 {alert['type']}: {alert.get('action', 'No action')}")

    # 3. Diagnostic final complet
    print("\n" + "=" * 80)
    print("🏁 DIAGNOSTIC FINAL COMPLET")
    print("=" * 80)

    final_analysis = diagnostics.analyze_convergence()
    diagnostics.print_summary(final_analysis)

    # 4. Générer les graphiques finaux
    print("\n📊 Génération des graphiques de diagnostic...")
    diagnostics.plot_diagnostics('diagnostic_final.png')

    # 5. Sauvegarder le rapport
    save_diagnostic_report(diagnostics, final_analysis)

    return diagnostics, final_analysis


if __name__ == "__main__":
    print("🎯 Démonstration du système de diagnostic de convergence")
    print("=" * 60)

    try:
        diagnostics, analysis = example_usage()

        print("\n✅ Démonstration terminée!")
        print("📁 Fichiers générés:")
        print("   • diagnostic_final.png - Graphiques de diagnostic")
        print("   • diagnostic_report.md - Rapport détaillé")
        print("\n💡 Pour intégrer à votre code:")
        print("   1. Utilisez DiagnosticCallbackManual dans votre boucle")
        print("   2. Appelez diagnostic_callback.on_epoch_end(epoch=i-1, logs=metrics)")
        print("   3. Utilisez RealTimeMonitor pour le monitoring par batch")

    except Exception as e:
        print(f"❌ Erreur lors de la démonstration: {e}")
        print("\n💡 Assurez-vous d'avoir installé:")
        print("   pip install matplotlib seaborn pandas numpy")

    print("\n🔧 Classes principales disponibles:")
    print("   • ConvergenceDiagnostics: Analyse complète de convergence")
    print("   • DiagnosticCallbackManual: Callback pour appel manuel")
    print("   • GoSpecificAnalysis: Analyses spécifiques au Go")
    print("   • RealTimeMonitor: Monitoring temps réel")