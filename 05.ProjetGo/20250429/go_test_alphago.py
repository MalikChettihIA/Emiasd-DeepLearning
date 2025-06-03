from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

from tensorflow.keras.models import load_model

EPOCHS = 500
N=10000

def test_alphago(nb_cycle, epochs=EPOCHS):

    model = load_model('alpha_zero_model.h5')
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.002,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=nb_cycle
    )
    return f'Model AlphaZero', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for nb_cycle in [1]:
        print(f'-------------------- Model AlphaZero -------------------- ')
        title, model, val, all_history, val_loss_history, total_time, lrs = test_alphago(nb_cycle)
        plot_learning_rate(lrs)

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