from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet
from go_mixnet import GoMixNet

EPOCHS = 250


def test_gomixnet(N, epochs=EPOCHS):

    model = GoMixNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=6, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N
    )
    return f'Model MixNet - N is {N}', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for N in [50000]:
        print(f'-------------------- Test GoMobileNet N {N} -------------------- ')
        mix_title, mix_model, mix_val, mix_all_history, mix_val_loss_history, mix_total_time, mix_lrs = test_gomixnet(N)
        plot_learning_rate(mix_lrs)

        # Affichage des résultats
        results = [
            (mix_model, mix_val, mix_title, mix_total_time)
        ]
        print_validation_results(results, epoch=EPOCHS)

        history.append(mix_all_history)
        loss_history.append(mix_val_loss_history)
        titles.append(mix_title)

    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=EPOCHS
    )