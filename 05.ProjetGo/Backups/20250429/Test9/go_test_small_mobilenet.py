from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

CONFIG = {
    'experiment_name': 'Test9',
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
        'initial_lr':0.0006, # Test 8 on passe de 0.0008 à 0.0004
        'clipnorm_lr':0.5, # Test 8 on passe de 1.0 à 0.5
        'first_decay_steps_lr': 0,
        't_mul_lr': 1.0,
        'm_mul_lr': 1.0,
        'nb_cosinedecay_cycle': 1,
        'policy_weight':1.0,
        'value_weight':1.0,
        'epochs':250,
        'N':25000
    },
    'augmentation':{
        'use_augmentation': True,
        'nb_rotation':8
    }
}

def test_gomobilenet(config):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=config['model']['block_num'] , filters=config['model']['filters'] , factor=config['model']['factor'],
                        se=config['model']['se'], drop_out_rate=config['model']['drop_out_rate'], activation=config['model']['activation'])
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