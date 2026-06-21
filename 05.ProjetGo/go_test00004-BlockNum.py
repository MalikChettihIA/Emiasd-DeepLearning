from go_test_utils import plot_version, print_validation_results, plot_learning_rate, plot_result, test_train_model

CONFIG = {
    'entity': 'Emiasd',
    'project': 'GoProject',
    'experiment_name': 'Test-BlockNum',
    'run':'0001',
    'model': {
        type:'GoMobileNet',
        'block_num': '??????????',
        'filters': 32,
        'factor': 4,
        'se':True,
        'drop_out_rate':0.3,
        'activation':'swish',
    },
    'train':  {
        'batch':32,
        'initial_lr':0.0006, # Test 8 on passe de 0.0008 à 0.0004
        'alpha_lr': 0.000006,
        'clipnorm_lr':0.5, # Test 10 on passe de  0.5 à 1.0
        'first_decay_steps_lr': 0,
        't_mul_lr': 1.0,
        'm_mul_lr': 1.0,
        'nb_cosinedecay_cycle': 1,
        'policy_weight':1.0,
        'value_weight':1.0,
        'epochs':100,
        'N':10000
    },
    'augmentation':{
        'use_augmentation': False,
        'nb_rotation':8
    }
}

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []

    for block_num in [3,5,7]:

        CONFIG['experiment_name'] = f'{CONFIG["experiment_name"]}-{N}'
        CONFIG['train']['block_num'] = block_num

        print(f'-------------------- {CONFIG["experiment_name"]}  -------------------- ')

        title, model, val, all_history, val_loss_history, total_time, lrs = test_train_model(config=CONFIG)
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