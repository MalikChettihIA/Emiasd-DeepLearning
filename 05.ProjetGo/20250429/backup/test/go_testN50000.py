from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

EPOCHS = 250

def test_gomobilenet(N, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N
    )
    return f'Model MobileNet - N is {N}', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for N in [25000]:
        print(f'-------------------- Test GoMobileNet N {N} -------------------- ')
        title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet(N)

        # Affichage des résultats

        results = [
            (model, val, title, total_time)
        ]
        print_validation_results(results, epoch=EPOCHS)

        history.append(all_history)
        loss_history.append(val_loss_history)
        titles.append(title)

    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=EPOCHS
    )