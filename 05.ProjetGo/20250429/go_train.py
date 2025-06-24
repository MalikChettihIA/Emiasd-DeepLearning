import time
import pandas as pd
import gc
import golois
import numpy as np

from tensorflow import keras
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecay, CosineDecayRestarts
from torch.ao.quantization.utils import activation_dtype


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

    Args:
        board: Plateau (19, 19, channels) - MODIFIÉ SUR PLACE
        policy: Politique - MODIFIÉ SUR PLACE
        transform_id: 0-7 pour les 8 transformations
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


def get_cosine_schedule_with_fixed_restarts(
        initial_lr=0.002,
        total_batches=200000,
        num_cycles=10,
):
    # Étapes par cycle (même longueur à chaque fois)
    first_decay_steps = total_batches // num_cycles

    cosine_lr = CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=1.0,  # cycles de même longueur
        m_mul=1.0,  # même amplitude
        alpha=initial_lr / 100  # lr final à zéro à la fin de chaque cycle
    )

    optimizer = optimizers.AdamW(learning_rate=cosine_lr, clipnorm=1.0)
    return optimizer


def train_model(model, batch=32, initial_lr=0.003, policy_weight=1.0, value_weight=1.0, epochs=100, N=10000,
                nb_cosinedecay_cycle=1, block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2,
                activation='swish', experiment_name=None, use_augmentation=True):
    # ✅ AJOUT: Setup WandB + Diagnostic + GradientDetector
    from wandb_integration import WandbGoTracker, WandbGoCallbackManual
    from convergence_diagnostics import ConvergenceDiagnostics, DiagnosticCallbackManual
    from gradient_detector import GradientVanishingDetector

    wandb_tracker = WandbGoTracker(
        project_name="go-mobilenet-training",
        experiment_name=f"{experiment_name} - mobilenet_block_num{block_num}_filters{filters}_e{epochs}_lr{initial_lr}_b{batch}_N{N}"
    )
    config = {
        "epochs": epochs, "batch_size": batch, "initial_lr": initial_lr,
        "policy_weight": policy_weight, "value_weight": value_weight,
        "cosine_cycles": nb_cosinedecay_cycle, "block_num": block_num, "filters": filters,
        "factor": factor, "se": se, "drop_out_rate": drop_out_rate, "activation": activation,
        "use_augmentation": use_augmentation  # 🔄 AJOUT: Info augmentation dans config
    }
    wandb_tracker.initialize_run(config=config, tags=["training"])
    wandb_tracker.log_model_architecture(model)

    diagnostics = ConvergenceDiagnostics()
    wandb_callback = WandbGoCallbackManual(wandb_tracker)
    wandb_callback.set_model(model)
    diagnostic_callback = DiagnosticCallbackManual(diagnostics)
    diagnostic_callback.set_model(model)

    detector = GradientVanishingDetector(model, wandb_tracker)
    detector.setup_gradient_tracking()

    # 🔄 AJOUT: Info augmentation
    if use_augmentation:
        print("🔄 Augmentation de données activée: 8 transformations par époque")
    else:
        print("❌ Entraînement sans augmentation de données")

    # ✅ FIN AJOUT

    start_time = time.time()

    # Configuration
    planes = 31
    moves = 361

    input_data = np.random.randint(2, size=(N, 19, 19, planes))
    input_data = input_data.astype('float32')

    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical(policy)

    value = np.random.randint(2, size=(N,))
    value = value.astype('float32')

    end = np.random.randint(2, size=(N, 19, 19, 2))
    end = end.astype('float32')

    groups = np.zeros((N, 19, 19, 1))
    groups = groups.astype('float32')

    # Get Validation Data
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    # Variable globale pour suivre la meilleure perte
    best_val_loss = float('inf')
    best_val_loss = 4.0
    logger = LrLogger()

    # get_cosine_schedule_with_fixed_restarts
    batches_per_epoch = N // batch  # 10000 // 32 ≈ 312

    # 🔄 AJOUT: Ajuster le nombre de batches si augmentation (x8 plus de données)
    if use_augmentation:
        batches_per_epoch *= 4  # 4 transformations = 8x plus de batches

    total_batches = batches_per_epoch * epochs  # ex: 312 * 640 ≈ 200000

    optimizer = get_cosine_schedule_with_fixed_restarts(
        initial_lr=initial_lr,
        total_batches=total_batches,
        num_cycles=nb_cosinedecay_cycle
    )

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

        # 🔄 AJOUT: Sauvegarder les données originales pour la boucle d'augmentation
        original_input = input_data.copy()
        original_policy = policy.copy()
        original_value = value.copy()

        # Variables pour accumuler les métriques de toutes les transformations
        epoch_histories = []
        transform_names = ["Original", "Rot90°", "Rot180°", "Rot270°",
                           "MirrorH", "MirrorV", "DiagMain", "DiagAnti"]

        # 🔄 BOUCLE D'AUGMENTATION INTÉGRÉE (0 à 7)
        augmentation_range = range(4) if use_augmentation else range(1)  # 8 transformations ou juste 1 (original)

        if use_augmentation:
            print(f"\n📊 Époque {i}/{epochs} - Entraînement sur 8 transformations:")

        for j in augmentation_range:
            transform_start_time = time.time()

            # Restaurer les données originales avant chaque transformation
            input_data[:] = original_input
            policy[:] = original_policy
            value[:] = original_value

            # Appliquer la transformation j à tout le batch (si j > 0)
            if j > 0:  # 0 = identité, pas besoin de transformer
                for sample_idx in range(N):
                    apply_go_transformation(input_data[sample_idx], policy[sample_idx], j)

            # Entraînement sur cette transformation
            history = model.fit(
                input_data,
                {'policy': policy, 'value': value},
                epochs=1,
                batch_size=batch,
                verbose=0 if use_augmentation and j > 0 else 1,  # Moins de verbosité pour les transformations
                callbacks=[logger]
            )

            # Enregistrer les métriques de cette transformation
            transform_metrics = {key: val[0] for key, val in history.history.items()}
            epoch_histories.append(transform_metrics)

            # Affichage pour chaque transformation si augmentation activée
            if use_augmentation:
                transform_time = time.time() - transform_start_time
                print(f"  🔀 {transform_names[j]:8s} (T{j}): "
                      f"loss={transform_metrics['loss']:.4f}, "
                      f"policy_loss={transform_metrics['policy_loss']:.4f}, "
                      f"value_loss={transform_metrics['value_loss']:.4f}, "
                      f"policy_categorical_accuracy={transform_metrics['policy_categorical_accuracy']:.4f}, "
                      f"value_mse={transform_metrics['value_mse']:.4f}, "
                      f"time={transform_time:.1f}s")

        # 🔄 AJOUT: Calculer les métriques moyennes de l'époque
        if use_augmentation:
            avg_metrics = {}
            for key in epoch_histories[0].keys():
                avg_metrics[key] = np.mean([h[key] for h in epoch_histories])
            metrics = avg_metrics
        else:
            metrics = {key: val[0] for key, val in history.history.items()}

        metrics['epoch'] = i
        all_history.append(metrics)

        # Dans votre boucle for (après model.fit)
        if i % 10 == 0:
            sample_inputs = input_data[:32]
            sample_policy = policy[:32]
            sample_value = value[:32]

            # ✅ Appel correct
            report = detector.comprehensive_check(
                inputs=sample_inputs,
                policy_targets=sample_policy,
                value_targets=sample_value,
                epoch=i
            )

            if report['health_status'] != 'HEALTHY':
                print(f"⚠️  Gradient {report['health_status']} at epoch {i}")
                for issue in report['critical_issues'][:2]:
                    print(f"   • {issue['type']}")

        # ✅ AJOUT: Appels manuels des callbacks
        wandb_callback.on_epoch_end(epoch=i - 1, logs=metrics)
        diagnostic_callback.on_epoch_end(epoch=i - 1, logs=metrics)

        # ✅ AJOUT: Log learning rate et timing
        timing_metrics = {
            "epoch_duration": time.time() - epoch_start_time,
            "total_elapsed": time.time() - start_time,
            "augmentation_enabled": use_augmentation  # 🔄 AJOUT: Track augmentation
        }
        wandb_callback.log_timing_manually(i, timing_metrics)

        if logger.lrs:
            wandb_callback.log_learning_rate_manually(i, logger.lrs[-1])
            diagnostic_callback.log_learning_rate_manually(i, logger.lrs[-1])
        # ✅ FIN AJOUT

        # 🔄 AJOUT: Affichage des métriques avec info augmentation
        epoch_time = time.time() - epoch_start_time
        aug_info = " (8 transforms avg)" if use_augmentation else ""
        print(
            f"\n✅ Époque {i}/{epochs}{aug_info}: time={epoch_time:.2f}s, "
            f"loss={metrics['loss']:.4f}, "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"value_loss={metrics['value_loss']:.4f}, "
            f"policy_categorical_accuracy={metrics['policy_categorical_accuracy']:.4f}, "
            f"value_mse={metrics['value_mse']:.4f}"
        )

        if i % 5 == 0:
            gc.collect()

        if i % 20 == 0:
            # 🔄 AJOUT: Restaurer les données originales pour la validation
            input_data[:] = original_input
            policy[:] = original_policy
            value[:] = original_value

            # Évaluation du modèle sur les données de validation (TOUJOURS sans augmentation)
            print("\n🔍 Évaluation sur données de validation (non augmentées)...")
            golois.getValidation(input_data, policy, value, end)
            val = model.evaluate(input_data, [policy, value], verbose=1, batch_size=batch)
            val_loss_history.append({
                'epoch': i,
                'val_policy_loss': val[1],
                'val_value_loss': val[2]
            })

            # ✅ AJOUT: Log validation vers WandB et Diagnostic
            val_metrics = {
                'val_total_loss': val[0],
                'val_policy_loss': val[1],
                'val_value_loss': val[2],
                'val_policy_categorical_accuracy': val[3],
                'val_value_mse': val[4]
            }
            wandb_callback.log_validation_manually(i, val_metrics)
            diagnostic_callback.log_validation_manually(i, val_metrics)
            # ✅ FIN AJOUT

            # Affichage des métriques
            print(
                f"📈 Validation Époque {i}/{epochs}: "
                f"loss={val[0]:.4f}, "
                f"policy_loss={val[1]:.4f}, "
                f"value_loss={val[2]:.4f}, "
                f"policy_categorical_accuracy={val[3]:.4f}, "
                f"value_mse={val[4]:.4f}"
            )
            current_val_loss = val[0]  # loss globale
            if current_val_loss < best_val_loss:
                # 🔄 AJOUT: Format propre du nom de fichier avec info augmentation
                aug_suffix = "_aug8x" if use_augmentation else "_no_aug"
                best_model_filename = f"best_model_epoch{i}_val{current_val_loss:.4f}{aug_suffix}.h5"
                best_model = model
                best_model_print = f"Best Model at epoch {i} : loss={val[0]}, policy_loss={val[1]:.4f}, value_loss={val[2]:.4f}"
                best_val_loss = current_val_loss
                best_model.save(best_model_filename)
                print(f"💾 Nouveau meilleur modèle sauvé : {best_model_filename}")

        # ✅ AJOUT: Vérification arrêt automatique si problème critique
        health_status = diagnostic_callback.get_current_health_status()
        if health_status == 'CRITICAL' and i > 50:
            print("\n🚨 TRAINING HALTED: Critical health status detected!")
            response = input("Continue training despite critical issues? (y/N): ")
            if response.lower() != 'y':
                print("🛑 Training stopped by user")
                break
        # ✅ FIN AJOUT

    # Fin FOR epoch
    print(f"\n🏆 {best_model_print}")
    if best_model_filename:
        best_model.save(best_model_filename)

    total_time = time.time() - start_time

    # 🔄 AJOUT: Statistiques finales d'augmentation
    if use_augmentation:
        total_samples = epochs * N * 8
        print(f"\n📊 Statistiques d'augmentation:")
        print(f"   🔄 8 transformations par époque")
        print(f"   📈 Échantillons par époque: {N} → {N * 8}")
        print(f"   🎯 Total d'échantillons traités: {total_samples:,}")
        print(f"   ⏱️  Temps total: {total_time:.2f}s")

    # ✅ AJOUT: Finalisation WandB et Diagnostic
    final_analysis = diagnostics.analyze_convergence()
    print("\n🏁 FINAL DIAGNOSTIC REPORT:")
    diagnostics.print_summary(final_analysis)

    wandb_tracker.log_training_diagnostics(final_analysis)

    final_metrics = {
        "final/total_training_time": total_time,
        "final/best_validation_loss": best_val_loss,
        "final/epochs_completed": i,
        "final/augmentation_used": use_augmentation  # 🔄 AJOUT: Log si augmentation utilisée
    }
    wandb_tracker.log_metrics(final_metrics)

    if logger.lrs:
        wandb_tracker.log_learning_rate_schedule(logger.lrs)

    wandb_tracker.finish_run()

    # ✅ AJOUT - bloc if avant return
    if detector.gradient_history:
        detector.plot_gradient_flow('gradient_analysis.png')
        final_stats = detector.gradient_history[-1]
        if final_stats.get('mean_gradient_norm', 0) < 1e-6:
            print("⚠️  GRADIENT VANISHING détecté")

    # ✅ FIN AJOUT

    return val, pd.DataFrame(all_history), pd.DataFrame(val_loss_history), total_time, logger.lrs