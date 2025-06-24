import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Dict, List, Optional, Tuple
import warnings


class GradientVanishingDetector:
    """
    Détecteur de gradient vanishing/exploding pour votre projet Go
    """

    def __init__(self, model: tf.keras.Model, wandb_tracker=None):
        self.model = model
        self.wandb_tracker = wandb_tracker
        self.gradient_history = []
        self.layer_gradient_norms = {}
        self.activation_history = {}
        self.weight_history = {}

        # Seuils de détection
        self.thresholds = {
            'vanishing_gradient': 1e-6,  # Gradients trop petits
            'exploding_gradient': 10.0,  # Gradients trop grands
            'dead_neurons': 0.1,  # % de neurones morts (ReLU saturés)
            'weight_change': 1e-8,  # Changement minimal des poids
            'activation_saturation': 0.95  # Saturation des activations
        }

    def setup_gradient_tracking(self):
        """Configure le tracking automatique des gradients"""

        # Hook pour capturer les gradients à chaque layer
        @tf.function
        def compute_gradients(inputs, policy_targets, value_targets):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)

                # Votre modèle retourne {'policy': policy_pred, 'value': value_pred}
                if isinstance(predictions, dict):
                    policy_pred = predictions['policy']
                    value_pred = predictions['value']
                else:
                    # Si c'est une liste [policy, value]
                    policy_pred, value_pred = predictions

                # Calcul des losses comme dans votre entraînement
                policy_loss = tf.keras.losses.categorical_crossentropy(policy_targets, policy_pred)
                value_loss = tf.keras.losses.mse(value_targets, tf.squeeze(value_pred))

                # Loss totale (vous pouvez ajuster les poids)
                total_loss = tf.reduce_mean(policy_loss) + tf.reduce_mean(value_loss)

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            return gradients, total_loss

        self.compute_gradients = compute_gradients

        print("✅ Gradient tracking configuré")

    def analyze_gradients(self, inputs, policy_targets, value_targets) -> Dict:
        """
        Analyse complète des gradients pour détecter vanishing/exploding
        """
        # Calcul des gradients
        gradients, loss = self.compute_gradients(inputs, policy_targets, value_targets)

        analysis = {
            'gradient_norms': [],
            'layer_analysis': {},
            'global_stats': {},
            'issues_detected': [],
            'recommendations': []
        }

        # 1. Analyse par couche
        for i, (grad, var) in enumerate(zip(gradients, self.model.trainable_variables)):
            if grad is not None:
                layer_name = var.name.split('/')[0] if '/' in var.name else f"layer_{i}"
                grad_norm = tf.norm(grad).numpy()
                analysis['gradient_norms'].append(grad_norm)

                # Analyse spécifique à cette couche
                layer_stats = {
                    'gradient_norm': grad_norm,
                    'gradient_mean': tf.reduce_mean(tf.abs(grad)).numpy(),
                    'gradient_std': tf.math.reduce_std(grad).numpy(),
                    'gradient_max': tf.reduce_max(tf.abs(grad)).numpy(),
                    'gradient_min': tf.reduce_min(tf.abs(grad)).numpy(),
                    'zero_gradients_ratio': tf.reduce_mean(
                        tf.cast(tf.abs(grad) < 1e-10, tf.float32)
                    ).numpy()
                }

                analysis['layer_analysis'][layer_name] = layer_stats

                # Détection de problèmes par couche
                if grad_norm < self.thresholds['vanishing_gradient']:
                    analysis['issues_detected'].append({
                        'type': 'GRADIENT_VANISHING',
                        'layer': layer_name,
                        'severity': 'HIGH' if grad_norm < 1e-8 else 'MEDIUM',
                        'gradient_norm': grad_norm
                    })

                elif grad_norm > self.thresholds['exploding_gradient']:
                    analysis['issues_detected'].append({
                        'type': 'GRADIENT_EXPLODING',
                        'layer': layer_name,
                        'severity': 'CRITICAL',
                        'gradient_norm': grad_norm
                    })

        # 2. Statistiques globales
        if analysis['gradient_norms']:
            analysis['global_stats'] = {
                'mean_gradient_norm': np.mean(analysis['gradient_norms']),
                'std_gradient_norm': np.std(analysis['gradient_norms']),
                'max_gradient_norm': np.max(analysis['gradient_norms']),
                'min_gradient_norm': np.min(analysis['gradient_norms']),
                'gradient_norm_ratio': np.max(analysis['gradient_norms']) / (np.min(analysis['gradient_norms']) + 1e-10)
            }

            # Détection de gradient vanishing global
            if analysis['global_stats']['mean_gradient_norm'] < self.thresholds['vanishing_gradient']:
                analysis['issues_detected'].append({
                    'type': 'GLOBAL_VANISHING',
                    'severity': 'HIGH',
                    'mean_norm': analysis['global_stats']['mean_gradient_norm']
                })

        # 3. Stocker l'historique
        self.gradient_history.append(analysis['global_stats'])

        return analysis

    def analyze_activations(self, inputs) -> Dict:
        """
        Analyse les activations pour détecter la saturation
        """
        # Créer un modèle pour extraire les activations intermédiaires
        layer_outputs = []
        for layer in self.model.layers:
            if hasattr(layer, 'activation') or 'conv' in layer.name.lower():
                layer_outputs.append(layer.output)

        if not layer_outputs:
            return {'warning': 'Aucune couche d\'activation trouvée'}

        activation_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=layer_outputs
        )

        activations = activation_model(inputs)
        if not isinstance(activations, list):
            activations = [activations]

        analysis = {
            'layer_activations': {},
            'saturation_issues': [],
            'dead_neurons': {}
        }

        for i, activation in enumerate(activations):
            layer_name = self.model.layers[i].name

            # Statistiques d'activation
            activation_stats = {
                'mean': tf.reduce_mean(activation).numpy(),
                'std': tf.math.reduce_std(activation).numpy(),
                'max': tf.reduce_max(activation).numpy(),
                'min': tf.reduce_min(activation).numpy(),
                'zeros_ratio': tf.reduce_mean(
                    tf.cast(activation == 0, tf.float32)
                ).numpy()
            }

            analysis['layer_activations'][layer_name] = activation_stats

            # Détection de neurones morts (ReLU saturés à 0)
            if activation_stats['zeros_ratio'] > self.thresholds['dead_neurons']:
                analysis['dead_neurons'][layer_name] = activation_stats['zeros_ratio']

            # Détection de saturation (activations trop près des bornes)
            if activation_stats['max'] > 0.95 or activation_stats['min'] < -0.95:
                analysis['saturation_issues'].append({
                    'layer': layer_name,
                    'type': 'ACTIVATION_SATURATION',
                    'max_activation': activation_stats['max'],
                    'min_activation': activation_stats['min']
                })

        return analysis

    def check_weight_updates(self, old_weights: List, new_weights: List) -> Dict:
        """
        Vérifie si les poids se mettent à jour correctement
        """
        analysis = {
            'weight_changes': [],
            'stagnant_layers': [],
            'update_ratios': []
        }

        for i, (old_w, new_w) in enumerate(zip(old_weights, new_weights)):
            weight_change = tf.norm(new_w - old_w).numpy()
            weight_norm = tf.norm(old_w).numpy()
            update_ratio = weight_change / (weight_norm + 1e-10)

            analysis['weight_changes'].append(weight_change)
            analysis['update_ratios'].append(update_ratio)

            # Détection de stagnation
            if weight_change < self.thresholds['weight_change']:
                analysis['stagnant_layers'].append({
                    'layer_index': i,
                    'weight_change': weight_change,
                    'update_ratio': update_ratio
                })

        return analysis

    def comprehensive_check(self, inputs, policy_targets, value_targets, epoch=None) -> Dict:
        """
        Vérification complète : gradients + activations + poids
        """
        print(f"\n🔍 Analyse complète gradient vanishing (époque {epoch})...")

        # 1. Sauvegarder les poids actuels
        current_weights = [w.numpy() for w in self.model.trainable_variables]

        # 2. Analyser les gradients
        gradient_analysis = self.analyze_gradients(inputs, policy_targets, value_targets)

        # 3. Analyser les activations
        activation_analysis = self.analyze_activations(inputs)

        # 4. Rapport consolidé
        report = {
            'epoch': epoch,
            'gradient_analysis': gradient_analysis,
            'activation_analysis': activation_analysis,
            'health_status': 'HEALTHY',
            'critical_issues': [],
            'recommendations': []
        }

        # 5. Évaluation de la santé globale
        critical_issues = []

        # Issues des gradients
        for issue in gradient_analysis['issues_detected']:
            if issue['severity'] in ['HIGH', 'CRITICAL']:
                critical_issues.append(issue)

        # Issues des activations
        if activation_analysis.get('dead_neurons'):
            for layer, ratio in activation_analysis['dead_neurons'].items():
                if ratio > 0.3:  # Plus de 30% de neurones morts
                    critical_issues.append({
                        'type': 'DEAD_NEURONS',
                        'layer': layer,
                        'dead_ratio': ratio,
                        'severity': 'HIGH'
                    })

        # Déterminer le statut de santé
        if critical_issues:
            if any(issue['severity'] == 'CRITICAL' for issue in critical_issues):
                report['health_status'] = 'CRITICAL'
            else:
                report['health_status'] = 'WARNING'

        report['critical_issues'] = critical_issues

        # 6. Générer des recommandations
        report['recommendations'] = self._generate_recommendations(critical_issues)

        # 7. Log vers WandB si disponible
        if self.wandb_tracker:
            self._log_to_wandb(report)

        return report

    def _generate_recommendations(self, issues: List[Dict]) -> List[str]:
        """Génère des recommandations basées sur les problèmes détectés"""
        recommendations = []

        issue_types = [issue['type'] for issue in issues]

        if 'GRADIENT_VANISHING' in issue_types:
            recommendations.extend([
                "🔧 Réduire la profondeur du réseau",
                "🔧 Utiliser l'activation Swish au lieu de ReLU",
                "🔧 Vérifier les connexions résiduelles (skip connections)",
                "🔧 Augmenter le learning rate initial",
                "🔧 Utiliser la normalisation par batch ou layer"
            ])

        if 'GRADIENT_EXPLODING' in issue_types:
            recommendations.extend([
                "🚨 URGENT: Réduire drastiquement le learning rate",
                "🔧 Implémenter gradient clipping (clipnorm=1.0)",
                "🔧 Vérifier la normalisation des données d'entrée",
                "🔧 Réduire la taille du batch"
            ])

        if 'DEAD_NEURONS' in issue_types:
            recommendations.extend([
                "🔧 Remplacer ReLU par Swish ou LeakyReLU",
                "🔧 Réduire le learning rate",
                "🔧 Vérifier l'initialisation des poids",
                "🔧 Ajouter de la normalisation"
            ])

        if 'GLOBAL_VANISHING' in issue_types:
            recommendations.extend([
                "🚨 ARCHITECTURE: Problème fondamental d'architecture",
                "🔧 Implémenter des connexions résiduelles",
                "🔧 Utiliser une architecture MobileNet avec skip connections",
                "🔧 Vérifier le bug potentiel dans _bottleneck_block"
            ])

        return recommendations

    def _log_to_wandb(self, report: Dict):
        """Log les métriques vers WandB"""
        if not self.wandb_tracker:
            return

        metrics = {}

        # Métriques des gradients
        if 'gradient_analysis' in report:
            grad_stats = report['gradient_analysis']['global_stats']
            metrics.update({
                'gradients/mean_norm': grad_stats.get('mean_gradient_norm', 0),
                'gradients/max_norm': grad_stats.get('max_gradient_norm', 0),
                'gradients/min_norm': grad_stats.get('min_gradient_norm', 0),
                'gradients/norm_ratio': grad_stats.get('gradient_norm_ratio', 0)
            })

        # Statut de santé
        health_mapping = {'HEALTHY': 1, 'WARNING': 0.5, 'CRITICAL': 0}
        metrics['gradient_health/status'] = health_mapping.get(report['health_status'], 0)
        metrics['gradient_health/num_issues'] = len(report['critical_issues'])

        # Neurones morts
        if 'activation_analysis' in report:
            dead_neurons = report['activation_analysis'].get('dead_neurons', {})
            if dead_neurons:
                metrics['activation/max_dead_ratio'] = max(dead_neurons.values())
                metrics['activation/avg_dead_ratio'] = np.mean(list(dead_neurons.values()))

        self.wandb_tracker.log_metrics(metrics, step=report.get('epoch'))

    def plot_gradient_flow(self, save_path: Optional[str] = None):
        """Visualise le flux des gradients à travers le réseau"""
        if not self.gradient_history:
            print("❌ Aucun historique de gradients disponible")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🔍 Analyse du Flux de Gradients', fontsize=16)

        # Historique des normes de gradients
        epochs = range(len(self.gradient_history))
        mean_norms = [h.get('mean_gradient_norm', 0) for h in self.gradient_history]
        max_norms = [h.get('max_gradient_norm', 0) for h in self.gradient_history]

        axes[0, 0].plot(epochs, mean_norms, label='Moyenne', color='blue')
        axes[0, 0].plot(epochs, max_norms, label='Maximum', color='red')
        axes[0, 0].axhline(y=self.thresholds['vanishing_gradient'],
                           color='orange', linestyle='--', label='Seuil vanishing')
        axes[0, 0].axhline(y=self.thresholds['exploding_gradient'],
                           color='red', linestyle='--', label='Seuil exploding')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Évolution des Normes de Gradients')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Norme (log scale)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Ratio des normes
        ratios = [h.get('gradient_norm_ratio', 0) for h in self.gradient_history]
        axes[0, 1].plot(epochs, ratios, color='green')
        axes[0, 1].set_title('Ratio Max/Min des Gradients')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].grid(True)

        # Statut de santé fictif pour démonstration
        axes[1, 0].text(0.5, 0.5, '📊 Dernière Analyse\n\n' +
                        f'Gradient moyen: {mean_norms[-1]:.2e}\n' +
                        f'Gradient max: {max_norms[-1]:.2e}\n' +
                        f'Ratio: {ratios[-1]:.2f}\n\n' +
                        ('✅ HEALTHY' if mean_norms[-1] > self.thresholds['vanishing_gradient']
                                        and max_norms[-1] < self.thresholds['exploding_gradient']
                         else '⚠️ PROBLÈME DÉTECTÉ'),
                        ha='center', va='center', transform=axes[1, 0].transAxes,
                        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')

        # Recommandations
        recommendations = [
            "• Surveiller les seuils critiques",
            "• Vérifier l'architecture si problème",
            "• Ajuster le learning rate",
            "• Utiliser gradient clipping si nécessaire"
        ]
        axes[1, 1].text(0.05, 0.95, '💡 Recommandations:\n\n' + '\n'.join(recommendations),
                        ha='left', va='top', transform=axes[1, 1].transAxes,
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Graphique sauvegardé: {save_path}")

        plt.show()


# ============================================================================
# CALLBACK KERAS POUR MONITORING AUTOMATIQUE
# ============================================================================

class GradientVanishingCallback(tf.keras.callbacks.Callback):
    """
    Callback pour monitoring automatique du gradient vanishing
    """

    def __init__(self, detector: GradientVanishingDetector,
                 check_frequency: int = 10,
                 sample_batch_size: int = 32):
        super().__init__()
        self.detector = detector
        self.check_frequency = check_frequency
        self.sample_batch_size = sample_batch_size
        self.sample_inputs = None
        self.sample_targets = None

    def on_train_begin(self, logs=None):
        self.detector.setup_gradient_tracking()
        print("🔍 Monitoring gradient vanishing activé")

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_frequency == 0 and self.sample_inputs is not None:
            report = self.detector.comprehensive_check(
                self.sample_inputs,
                self.sample_targets,
                epoch
            )

            if report['health_status'] != 'HEALTHY':
                print(f"\n⚠️  ALERTE GRADIENTS (Époque {epoch}): {report['health_status']}")
                for issue in report['critical_issues']:
                    print(f"   🚨 {issue['type']} - {issue.get('layer', 'Global')}")

                print("💡 Recommandations:")
                for rec in report['recommendations'][:3]:  # Top 3
                    print(f"   {rec}")

    def on_batch_end(self, batch, logs=None):
        # Capturer un échantillon pour l'analyse
        if self.sample_inputs is None and hasattr(self.model, '_current_batch_data'):
            # Vous devrez adapter cette partie selon votre structure de données
            pass

