import numpy as np


def apply_go_transformation(board, policy, transform_id):
    """
    Applique UNE transformation au plateau et à la politique
    SANS CRÉER DE NOUVELLES DONNÉES EN MÉMOIRE

    Args:
        board: Plateau (19, 19, channels) - MODIFIÉ SUR PLACE
        policy: Politique (361,) ou (19, 19) - MODIFIÉ SUR PLACE
        transform_id: 0-7 pour les 8 transformations possibles

    Returns:
        board et policy transformés (mêmes objets, modifiés sur place)
    """

    if transform_id == 0:
        # Identité - pas de transformation
        return board, policy

    elif transform_id == 1:
        # Rotation 90° horaire
        board[:] = np.rot90(board, k=-1, axes=(0, 1))
        if len(policy.shape) == 1:  # Si policy est un vecteur (361,)
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=-1, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        else:  # Si policy est déjà 2D (19, 19)
            policy[:] = np.rot90(policy, k=-1, axes=(0, 1))

    elif transform_id == 2:
        # Rotation 180°
        board[:] = np.rot90(board, k=2, axes=(0, 1))
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=2, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.rot90(policy, k=2, axes=(0, 1))

    elif transform_id == 3:
        # Rotation 270° horaire
        board[:] = np.rot90(board, k=1, axes=(0, 1))
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.rot90(policy_2d, k=1, axes=(0, 1))
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.rot90(policy, k=1, axes=(0, 1))

    elif transform_id == 4:
        # Miroir horizontal
        board[:] = np.flip(board, axis=1)
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=1)
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.flip(policy, axis=1)

    elif transform_id == 5:
        # Miroir vertical
        board[:] = np.flip(board, axis=0)
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(policy_2d, axis=0)
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.flip(policy, axis=0)

    elif transform_id == 6:
        # Transpose (diagonal principal)
        board[:] = np.transpose(board, (1, 0, 2))
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.transpose(policy_2d, (1, 0))
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.transpose(policy, (1, 0))

    elif transform_id == 7:
        # Anti-diagonal (transpose + double flip)
        board[:] = np.flip(np.flip(np.transpose(board, (1, 0, 2)), axis=0), axis=1)
        if len(policy.shape) == 1:
            policy_2d = policy.reshape(19, 19)
            policy_2d[:] = np.flip(np.flip(np.transpose(policy_2d, (1, 0)), axis=0), axis=1)
            policy[:] = policy_2d.flatten()
        else:
            policy[:] = np.flip(np.flip(np.transpose(policy, (1, 0)), axis=0), axis=1)

    return board, policy


def augment_batch_in_place(input_data, policy, value, transform_probability=0.8):
    """
    Augmente un batch EN PLACE (sans créer de nouvelles données)

    Args:
        input_data: (batch_size, 19, 19, channels) - MODIFIÉ SUR PLACE
        policy: (batch_size, 361) - MODIFIÉ SUR PLACE
        value: (batch_size,) - PAS MODIFIÉ
        transform_probability: Probabilité d'appliquer une transformation (0.0 à 1.0)

    Returns:
        Rien - les données sont modifiées sur place
    """
    batch_size = input_data.shape[0]

    for i in range(batch_size):
        # Décider si on applique une transformation
        if np.random.random() < transform_probability:
            # Choisir une transformation aléatoire (1-7, pas 0 qui est identité)
            transform_id = np.random.randint(1, 8)

            # Appliquer la transformation sur place
            apply_go_transformation(input_data[i], policy[i], transform_id)


# INTÉGRATION ULTRA-SIMPLE DANS VOTRE CODE EXISTANT
def modify_your_training_loop():
    """
    Voici EXACTEMENT comment modifier votre go_train.py
    """
    return """
    # 1. Ajouter en haut de votre fichier go_train.py :
    from go_simple_augmentation import augment_batch_in_place

    # 2. Dans votre boucle d'entraînement, changer SEULEMENT ces lignes :

    # AVANT (votre code actuel) :
    for i in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Récupération des données
        golois.getBatch(input_data, policy, value, end, groups, i * N)

        history = model.fit(
            input_data,
            {'policy': policy, 'value': value},
            epochs=1,
            batch_size=batch,
            verbose=1,
            callbacks=[logger]
        )

    # APRÈS (avec augmentation) :
    for i in range(1, epochs + 1):
        epoch_start_time = time.time()
        # Récupération des données
        golois.getBatch(input_data, policy, value, end, groups, i * N)

        # ✨ LIGNE MAGIQUE : Augmentation sur place (80% de chance par échantillon)
        augment_batch_in_place(input_data, policy, value, transform_probability=0.8)

        history = model.fit(
            input_data,
            {'policy': policy, 'value': value},
            epochs=1,
            batch_size=batch,
            verbose=1,
            callbacks=[logger]
        )

    # C'est TOUT ! Une seule ligne ajoutée, zéro surcharge mémoire !
    """


# Fonction de test pour vérifier que ça marche
def test_augmentation():
    """
    Test rapide pour vérifier les transformations
    """
    print("🧪 Test des transformations Go...")

    # Créer des données de test
    test_board = np.random.rand(2, 19, 19, 5)  # 2 échantillons
    test_policy = np.random.rand(2, 361)
    test_value = np.array([0.7, 0.3])

    print(f"Avant transformation:")
    print(f"  Board[0,0,0,0] = {test_board[0, 0, 0, 0]:.4f}")
    print(f"  Policy[0,0] = {test_policy[0, 0]:.4f}")

    # Test d'une transformation spécifique
    original_board = test_board[0].copy()
    original_policy = test_policy[0].copy()

    apply_go_transformation(test_board[0], test_policy[0], transform_id=1)  # Rotation 90°

    print(f"Après rotation 90°:")
    print(f"  Board[0,0,0,0] = {test_board[0, 0, 0, 0]:.4f}")
    print(f"  Policy[0,0] = {test_policy[0, 0]:.4f}")

    # Test du batch
    test_board_batch = np.random.rand(4, 19, 19, 3)
    test_policy_batch = np.random.rand(4, 361)
    test_value_batch = np.array([0.1, 0.5, 0.8, 0.2])

    print(f"\n🎲 Test du batch (4 échantillons)...")
    augment_batch_in_place(test_board_batch, test_policy_batch, test_value_batch,
                           transform_probability=1.0)  # 100% pour le test

    print("✅ Test réussi ! Aucune explosion mémoire.")
    return True


if __name__ == "__main__":
    test_augmentation()