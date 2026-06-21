 # Import Libraries

import sys
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from go_train import GoTrainer
from go_mobilenet import GoMobileNet
from go_mobilenetv2 import GoMobileNetV2
from go_mixnet import GoMixNet
from go_resnet import GoResNet

def plot_version():
    print ("Python version", sys.version_info)
    print ("Tensorflow version", tf.__version__)

def test_train_model(config):

    model = get_model(config)
    #if config['model']['name'] is None:
    #    model = get_model(config)
    #else:
    #    model = load_model(config['model']['name'])

    trainer = GoTrainer(config)
    val, all_history, val_loss_history, total_time, lrs = trainer.train(
        model
    )
    return (f"{config['experiment_name']}"), model, val, all_history, val_loss_history, total_time, lrs

def get_model(config):

    if config['model']['type'] == 'GoMixNet':
        model = get_gomixnet_model(config)
    elif config['model']['type'] == 'GoResNet':
        model = get_goresnet_model(config)
    elif config['model']['type'] == 'GoMobileNet':
        model = get_gomobilenet_model(config)
    elif config['model']['type'] == 'GoMobileNetV2':
        model = get_gomobilenetv2_model(config)
    else :
        model = get_gomobilenet_model(config)


    model = model.build(block_num=config['model']['block_num'], filters=config['model']['filters'],
                        factor=config['model']['factor'],
                        se=config['model']['se'], drop_out_rate=config['model']['drop_out_rate'],
                        activation=config['model']['activation'],
                        kernel_size=config['model']['kernel_size'],
                        regul_l2=config['model']['regul_l2'])
    model.summary()
    return model

def get_goresnet_model(config):
    config['experiment_name'] = f'{config["experiment_name"]}'
    config['model']['block_num'] = 5
    model = GoResNet((19, 19, 31), 361)
    return model

def get_gomobilenet_model(config):
    config['experiment_name'] = f'{config["experiment_name"]}'
    config['model']['block_num'] = 7
    model = GoMobileNet((19, 19, 31), 361)
    return model

def get_gomobilenetv2_model(config):
    config['experiment_name'] = f'{config["experiment_name"]}'
    config['model']['block_num'] = 9
    model = GoMobileNetV2((19, 19, 31), 361)
    return model

def get_gomixnet_model(config):
    config['experiment_name'] = f'{config["experiment_name"]}'
    config['model']['block_num'] = 6
    model = GoMixNet((19, 19, 31), 361)
    return model

def plot_learning_rate(lrs):
    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.title('Learning Rate Evolution During Training')
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.show()
    
def print_validation_results(model_results, epoch=100):
    for model, val, label, time in model_results:
        metrics = dict(zip(model.metrics_names, val))
        title = f"📊 Validation Results for {label}"
        if epoch is not None:
            title += f" — Epoch {epoch}"
        print(f"\n{title}:")
        for name, value in metrics.items():
            print(f"  - {name:<30}: {value:.4f}")
        print(f"  - Time: {time:.4f}")

def plot_result(history_dfs, val_dfs, labels, epochs=None):
    assert len(history_dfs) == len(labels)

    title = f"Epochs: {epochs}"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 3)

    # --- Ligne 1 : 3 plots ---
    ax1 = fig.add_subplot(gs[0, 0])
    for df, val_df, label in zip(history_dfs, val_dfs, labels):
        ax1.plot(df['epoch'], df['loss'], label=f'{label} Train Loss')
        if val_df is not None:
            if 'val_policy_loss' in val_df.columns and 'val_value_loss' in val_df.columns:
                val_total_loss = val_df['val_policy_loss'] + val_df['val_value_loss']
                ax1.plot(val_df['epoch'], val_total_loss, 'o--', label=f'{label} Val Loss (recalculated)')
    ax1.set_title('Total Loss par Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    for df, val_df, label in zip(history_dfs, val_dfs, labels):
        if 'policy_loss' in df.columns:
            ax2.plot(df['epoch'], df['policy_loss'], label=f'{label} Train Policy Loss')
        if val_df is not None and 'val_policy_loss' in val_df.columns:
            ax2.plot(val_df['epoch'], val_df['val_policy_loss'], 'o--', label=f'{label} Val Policy Loss')
    ax2.set_title('Policy Loss par Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Policy Loss')
    ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    for df, val_df, label in zip(history_dfs, val_dfs, labels):
        if 'value_loss' in df.columns:
            ax3.plot(df['epoch'], df['value_loss'], label=f'{label} Train Value Loss')
        if val_df is not None and 'val_value_loss' in val_df.columns:
            ax3.plot(val_df['epoch'], val_df['val_value_loss'], 'o--', label=f'{label} Val Value Loss')
    ax3.set_title('Value Loss par Epoch')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Value Loss')
    ax3.legend()

    # --- Ligne 2 : 2 plots ---
    # --- Policy Accuracy (avec validation) ---
    ax4 = fig.add_subplot(gs[1, 0])
    for df, val_df, label in zip(history_dfs, val_dfs, labels):
        if 'policy_categorical_accuracy' in df.columns:
            ax4.plot(df['epoch'], df['policy_categorical_accuracy'], label=f'{label} Train Policy Acc')
        if val_df is not None and 'val_policy_accuracy' in val_df.columns:
            ax4.plot(val_df['epoch'], val_df['val_policy_accuracy'], 'o--', label=f'{label} Val Policy Acc')
    ax4.set_title('Policy Accuracy par Epoch')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Categorical Accuracy')
    ax4.legend()

    # --- Value MSE (avec validation) ---
    ax5 = fig.add_subplot(gs[1, 1])
    for df, val_df, label in zip(history_dfs, val_dfs, labels):
        if 'value_mse' in df.columns:
            ax5.plot(df['epoch'], df['value_mse'], label=f'{label} Train Value MSE')
        if val_df is not None and 'val_value_mse' in val_df.columns:
            ax5.plot(val_df['epoch'], val_df['val_value_mse'], 'o--', label=f'{label} Val Value MSE')
    ax5.set_title('Value MSE par Epoch')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('MSE')
    ax5.legend()

    # Libérer la dernière case vide
    fig.delaxes(fig.add_subplot(gs[1, 2]))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()