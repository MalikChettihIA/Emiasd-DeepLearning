from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet
from go_mixnet import GoMixNet

EPOCHS = 100

def test_gomobilenet1(nb_cycle, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=7, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.002,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=10000,
        nb_cosinedecay_cycle=nb_cycle
    )
    return f'Model GoMobileNet1 ({model.count_params()}) - block_num=7, filters=32, factor=4, se=True, drop_out_rate=0.2, activation=swish', model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet2(nb_cycle, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.002,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=10000,
        nb_cosinedecay_cycle=nb_cycle
    )
    return f'Model GoMobileNet2 ({model.count_params()}) - block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2, activation=swish', model, val, all_history, val_loss_history, total_time, lrs


if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for nb_cycle in [1]:
        print(f'-------------------- Model GoMixNet - Nb CosineDecay Cycle is {nb_cycle} -------------------- ')
        title1, model1, val1, all_history1, val_loss_history1, total_time1, lrs1 = test_gomobilenet1(nb_cycle)
        title2, model2, val2, all_history2, val_loss_history2, total_time2, lrs2 = test_gomobilenet2(nb_cycle)

        # Affichage des résultats
        results = [
            (model1, val1, title1, total_time1),
            (model2, val2, title2, total_time2)
        ]
        print_validation_results(results, epoch=EPOCHS)

        history.append(all_history1)
        history.append(all_history2)
        loss_history.append(val_loss_history1)
        loss_history.append(val_loss_history2)
        titles.append(title1)
        titles.append(title2)

    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=EPOCHS
    )