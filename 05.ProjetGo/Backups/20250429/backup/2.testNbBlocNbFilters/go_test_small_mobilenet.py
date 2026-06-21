from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

EPOCHS = 100
N=10000

def test_gomobilenet( epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=2 , filters=64 , factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.02,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=1,
        block_num=2, filters=64, factor=4, se=True, drop_out_rate=0.2, activation='swish'
    )
    return (f'Model GoMobileNet block_num=2 , filters=64'), model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet2( epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=8 , filters=32 , factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.02,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=1,
        block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish'
    )
    return (f'Model GoMobileNet block_num=8 , filters=32'), model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet3( epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=6 , filters=32 , factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.02,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=1,
        block_num=6, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish'
    )
    return (f'Model GoMobileNet block_num=6 , filters=32'), model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet4( epochs=EPOCHS):

    model = GoMobileNet((19, 19, 31), 361)
    model = model.build(block_num=4 , filters=32 , factor=4, se=True, drop_out_rate=0.2, activation='swish')
    model.summary()

    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.02,
        policy_weight=1.0,
        value_weight=1.0,
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=1,
        block_num=4, filters=32, factor=4, se=True, drop_out_rate=0.2, activation='swish'
    )
    return (f'Model GoMobileNet block_num=4 , filters=32 '), model, val, all_history, val_loss_history, total_time, lrs

if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []

    print(f'-------------------- Model GoMobileNet - Nb CosineDecay Cycle is  -------------------- ')
    title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet()
    plot_learning_rate(lrs)

    # Affichage des résultats
    results = [
        (model, val, title, total_time)
    ]
    print_validation_results(results, epoch=EPOCHS)

    history.append(all_history)
    loss_history.append(val_loss_history)
    titles.append(title)

    title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet2()
    plot_learning_rate(lrs)

    # Affichage des résultats
    results = [
        (model, val, title, total_time)
    ]
    print_validation_results(results, epoch=EPOCHS)

    history.append(all_history)
    loss_history.append(val_loss_history)
    titles.append(title)

    title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet3()
    plot_learning_rate(lrs)

    # Affichage des résultats
    results = [
        (model, val, title, total_time)
    ]
    print_validation_results(results, epoch=EPOCHS)

    history.append(all_history)
    loss_history.append(val_loss_history)
    titles.append(title)

    title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet4()
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