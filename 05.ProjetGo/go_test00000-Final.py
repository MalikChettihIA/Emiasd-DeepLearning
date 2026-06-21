from go_test_utils import test_train_model, plot_version, print_validation_results, plot_learning_rate, plot_result


CONFIG = {
    'entity': 'Emiasd',
    'project': 'GoProject',
    'experiment_name': 'Test-FinalMixNet',
    'model': {
        'type':'GoMixNet',
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
        'batch':64,
        'initial_lr':0.0006,
        'alpha_lr': 0.000006,
        'clipnorm_lr':0.5,
        'first_decay_steps_lr': 0,
        't_mul_lr': 1.0,
        'm_mul_lr': 1.0,
        'nb_cosinedecay_cycle': 1,
        'policy_weight':1.0,
        'value_weight':4.0,
        'epochs':300,
        'N':25000
    },
    'augmentation':{
        'use_augmentation': True,
        'mode': 'all_transforms_per_batch',
        'nb_rotation': 8,
        'transform_probability': 1.0
    }
}


if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []

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