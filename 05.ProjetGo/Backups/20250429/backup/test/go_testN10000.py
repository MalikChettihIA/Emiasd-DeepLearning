from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet
from go_mixnet import GoMixNet

EPOCHS = 50

def test_gomobilenet(lr, epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=17, filters=128, factor=4, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=lr,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=10000
    )
    return f'Model MobileNet - LR is {lr}', model, val, all_history, val_loss_history, total_time, lrs

def test_gomixnet(lr, epochs=EPOCHS):

    model = GoMixNet((19, 19, 31), 361)
    model = model.build(block_num=17, filters=128, factor=6, se=True, drop_out_rate=0.5, activation='swish')
    model.summary()
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=lr,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=10000
    )
    return f'Model MixNet - LR is {N}', model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []
    for lr in [0.1, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.000]:
        print(f'-------------------- Test GoMobileNet N {lr} -------------------- ')
        title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet(lr)
        #mix_title, mix_model, mix_val, mix_all_history, mix_val_loss_history, mix_total_time, mix_lrs = test_gomixnet(N)
        plot_learning_rate(lrs)
        #plot_learning_rate(mix_lrs)

        # Affichage des résultats
        results = [
            (model, val, title, total_time)#,
            #(mix_model, mix_val, mix_title, mix_total_time)
        ]
        print_validation_results(results, epoch=EPOCHS)

        history.append(all_history)
        #history.append(mix_all_history)
        loss_history.append(val_loss_history)
        #loss_history.append(mix_val_loss_history)
        titles.append(title)
        #titles.append(mix_title)

    # Affichage des courbes comparatives
    plot_result(
        history_dfs=history,
        val_dfs=loss_history,
        labels=titles,
        epochs=EPOCHS
    )