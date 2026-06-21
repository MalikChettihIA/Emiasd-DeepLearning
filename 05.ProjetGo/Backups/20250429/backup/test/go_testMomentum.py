from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet
from go_mixnet import GoMixNet

EPOCHS = 100

def test_momentum9(N, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        momentum=0.9,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N
    )
    return f'Model MobileNet - Momentum 0.9', model, val, all_history, val_loss_history, total_time, lrs


def test_momentum99(N, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        momentum=0.99,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N
    )
    return f'Model MobileNet - Momentum 0.99', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for N in [10000]:
        print(f'-------------------- Test GoMobileNet N {N} -------------------- ')
        title9, model9, val9, all_history9, val_loss_history9, total_time9, lrs9 = test_momentum9(N)
        title99, model99, val99, all_history99, val_loss_history99, total_time99, lrs99 = test_momentum99(N)
        plot_learning_rate(lrs9)
        plot_learning_rate(lrs99)

        # Affichage des résultats
        results = [
            (model9, val9, title9, total_time9),
            (model99, val99, title99, total_time99)
        ]
        print_validation_results(results, epoch=EPOCHS)

        history.append(all_history9)
        history.append(all_history99)
        loss_history.append(val_loss_history9)  
        loss_history.append(val_loss_history99)
        titles.append(title9)
        titles.append(title99)
        
    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=EPOCHS
    )