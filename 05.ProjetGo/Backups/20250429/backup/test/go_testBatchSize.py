from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

EPOCHS = 250

def test_gomobilenet(batchsize, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=batchsize,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=10000
    )
    return f'Model MobileNet - BatchSize is {batchsize}', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    all_bacthes_history = []
    all_bacthes_val_loss_history = []
    all_bacthes_titles_history = []
    for batchsize in [32, 64, 128, 256, 512]:
        print(f'-------------------- Test GoMobileNet Batch {batchsize} -------------------- ')
        title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet(batchsize)

        # Affichage des résultats

        results = [
            (model, val, title, total_time)
        ]
        print_validation_results(results, epoch=EPOCHS)

        all_bacthes_history.append(all_history)
        all_bacthes_val_loss_history.append(val_loss_history)
        all_bacthes_titles_history.append(title)

    # Affichage des courbes comparatives
    plot_result(
        history_dfs=all_bacthes_history,
        val_dfs=all_bacthes_val_loss_history,
        labels=all_bacthes_titles_history,
        epochs=EPOCHS
    )