from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from tensorflow import keras

CONFIG = {
    'experiment_name': 'Test10-02-iter02'
                       '',
    'model': {
        'block_num': 7,
        'filters': 32,
        'factor': 4,
        'se':True,
        'drop_out_rate':0.3,
        'activation':'swish',
    },
    'train':  {
        'batch':64,
        'initial_lr':0.0005, # Test 8 on passe de 0.0008 à 0.0004
        'alpha_lr': 0.00005,
        'clipnorm_lr':0.5, # Test 10 on passe de  0.5 à 1.0
        'first_decay_steps_lr': 0,
        't_mul_lr': 1.0,
        'm_mul_lr': 1.0,
        'nb_cosinedecay_cycle': 1,
        'policy_weight':1.0,
        'value_weight':1.0,
        'epochs':50,
        'N':10000
    },
    'augmentation':{
        'use_augmentation': True,
        'nb_rotation':8
    }
}

def test_gomobilenet(config):
    model_path = 'best_model_epoch240_val2.5550_aug8x.h5'

    model = keras.models.load_model(model_path)
    print(f"Modèle chargé depuis: {model_path}")
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        config=config
    )
    return (f"Model GoMobileNet {config['experiment_name']}"), model, val, all_history, val_loss_history, total_time, lrs


if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []

    print(f'-------------------- Model GoMobileNet - Nb CosineDecay Cycle is  -------------------- ')
    title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet(config=CONFIG)
    plot_learning_rate(lrs)

    # Affichage des résultats
    results = [
        (model, val, title, total_time)
    ]
    print_validation_results(results, epoch=CONFIG['train']['epochs'])

    history.append(all_history)
    loss_history.append(val_loss_history)
    titles.append(title)


    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=CONFIG['train']['epochs']
    )