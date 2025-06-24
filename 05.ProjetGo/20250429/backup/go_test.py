# go_test_small_mobilenetv2.py - Version corrigée

from go_utils import plot_version, print_validation_results, plot_learning_rate, plot_result
from go_train import train_model
from go_mobilenet import GoMobileNet

EPOCHS = 100
N = 10000


def test_gomobilenet(nb_cycle=1, epochs=EPOCHS):
    """Test avec buildV2 corrigé"""

    # Création du modèle
    model = GoMobileNet((19, 19, 31), 361)

    # ✅ Utilisation de buildV2 avec paramètres corrects
    model = model.build(
        block_num=8,
        filters=32,  # ✅ Correction: 32 au lieu de 31
        factor=4,
        se=True,
        drop_out_rate=0.3,
        activation='swish'
    )

    print("📋 Architecture du modèle:")
    model.summary()

    # Entraînement avec paramètres optimisés pour features spécialisées
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.005,  # ✅ LR plus élevé comme recommandé
        policy_weight=1.0,
        value_weight=0.5,  # ✅ Poids value réduit initialement
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=nb_cycle,
        block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.3, activation='swish', experiment_name='Test3'
    )

    return (
        f'Model GoMobileNet V2 - blocks={8}, filters={32}, LR={0.005}'), model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet2(nb_cycle=1, epochs=EPOCHS):
    """Test avec buildV2 corrigé"""

    # Création du modèle
    model = GoMobileNet((19, 19, 31), 361)

    # ✅ Utilisation de buildV2 avec paramètres corrects
    model = model.buildV2(
        block_num=8,
        filters=32,  # ✅ Correction: 32 au lieu de 31
        factor=4,
        se=True,
        drop_out_rate=0.3,
        activation='swish'
    )

    print("📋 Architecture du modèle:")
    model.summary()

    # Entraînement avec paramètres optimisés pour features spécialisées
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.001,  # ✅ LR plus élevé comme recommandé
        policy_weight=1.0,
        value_weight=0.5,  # ✅ Poids value réduit initialement
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=nb_cycle,
        block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.3, activation='swish', experiment_name='Test3'
    )

    return (
        f'Model GoMobileNet V2 - blocks={8}, filters={32}, LR={0.005}'), model, val, all_history, val_loss_history, total_time, lrs

def test_gomobilenet3(nb_cycle=1, epochs=EPOCHS):
    """Test avec buildV2 corrigé"""

    # Création du modèle
    model = GoMobileNet((19, 19, 31), 361)

    # ✅ Utilisation de buildV2 avec paramètres corrects
    model = model.buildV3(
        block_num=8,
        filters=32,  # ✅ Correction: 32 au lieu de 31
        factor=4,
        se=True,
        drop_out_rate=0.3,
        activation='swish'
    )

    print("📋 Architecture du modèle:")
    model.summary()

    # Entraînement avec paramètres optimisés pour features spécialisées
    val, all_history, val_loss_history, total_time, lrs = train_model(
        model,
        batch=32,
        initial_lr=0.001,  # ✅ LR plus élevé comme recommandé
        policy_weight=1.0,
        value_weight=0.5,  # ✅ Poids value réduit initialement
        epochs=epochs,
        N=N,
        nb_cosinedecay_cycle=nb_cycle,
        block_num=8, filters=32, factor=4, se=True, drop_out_rate=0.3, activation='swish', experiment_name='Test3'
    )

    return (
        f'Model GoMobileNet V2 - blocks={8}, filters={32}, LR={0.005}'), model, val, all_history, val_loss_history, total_time, lrs
if __name__ == "__main__":

    plot_version()

    history = []
    loss_history = []
    titles = []

    for nb_cycle in [1]:

        try:
            title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet2(nb_cycle)

            plot_learning_rate(lrs)

            # Affichage des résultats
            results = [(model, val, title, total_time)]
            print_validation_results(results, epoch=EPOCHS)

            history.append(all_history)
            loss_history.append(val_loss_history)
            titles.append(title)

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            print("🔧 Vérifiez que buildV2 est correctement défini dans go_mobilenet.py")
            break

        try:
            title, model, val, all_history, val_loss_history, total_time, lrs = test_gomobilenet3(nb_cycle)

            plot_learning_rate(lrs)

            # Affichage des résultats
            results = [(model, val, title, total_time)]
            print_validation_results(results, epoch=EPOCHS)

            history.append(all_history)
            loss_history.append(val_loss_history)
            titles.append(title)

        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            print("🔧 Vérifiez que buildV2 est correctement défini dans go_mobilenet.py")
            break

    # Affichage des courbes comparatives
    if history:
        plot_result(
            history_dfs=history,
            val_dfs=loss_history,
            labels=titles,
            epochs=EPOCHS
        )
    else:
        print("⚠️  Aucun résultat à afficher - problème de configuration")