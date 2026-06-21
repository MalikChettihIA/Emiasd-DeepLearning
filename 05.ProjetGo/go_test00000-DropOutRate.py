from go_test_utils import test_train_model, plot_version, print_validation_results, plot_learning_rate, plot_result


CONFIG = {
    'entity': 'Emiasd',
    'project': 'GoProject',
    'experiment_name': 'Test-DropOutRate',
    'model': {
        'type':'GoMobilNet',
        'block_num': 7,
        'kernel_size': (3,3),
        'filters': 32,
        'factor': 4,
        'se':True,
        'drop_out_rate':0.3,
        'activation':'swish',
        'regul_l2': 0.0001
    },
    'train':  {
        'batch':32,
        'initial_lr':0.0006,
        'alpha_lr': 0.000006,
        'clipnorm_lr':0.5,
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
        'use_augmentation': False,
        'nb_rotation':8
    }
}


if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    experiment_name = CONFIG['experiment_name']
    for drop_out_rate in [0.2, 0.3, 0.5]:
        print(f'-------------------- {CONFIG["experiment_name"]}  -------------------- ')

        CONFIG['train']['drop_out_rate'] = drop_out_rate
        CONFIG['experiment_name'] = f'{experiment_name}-{drop_out_rate}'
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