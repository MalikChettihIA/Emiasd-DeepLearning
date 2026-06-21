import os
import numpy as np
import golois
from tensorflow.keras.models import load_model

if __name__ == "__main__":
    # Paramètres
    N = 10000  # ou 5000 si mémoire limitée
    planes = 31
    moves = 361

    # Tenseurs d'entrée/sortie
    input_data = np.zeros((N, 19, 19, planes), dtype='float32')
    policy = np.zeros((N, moves), dtype='float32')
    value = np.zeros((N,), dtype='float32')
    end = np.zeros((N, 19, 19, 2), dtype='float32')  # probablement non utilisé

    # ❗ Générer les données UNE FOIS pour tous les modèles
    golois.getValidation(input_data, policy, value, end)

    best_model_name = None
    best_total_loss = float('inf')
    best_results = None
    best_model = None

    # Boucle sur les modèles .h5 dans le répertoire
    for filename in os.listdir(""):
        if filename.endswith(".h5"):
            try:
                print(f"\nÉvaluation du modèle : {filename}")
                model = load_model(filename)
                results = model.evaluate(input_data, [policy, value], verbose=1)
                print(f" - Total loss     : {results[0]:.4f}")
                print(f" - Policy loss    : {results[1]:.4f}")
                print(f" - Value loss     : {results[2]:.4f}")
                print(f" - Policy accuracy: {results[3]:.4f}")
                print(f" - Value accuracy : {results[4]:.4f}")

                if results[0] < best_total_loss:
                    best_total_loss = results[0]
                    best_model_name = filename
                    best_results = results
                    best_model = model
            except Exception as e:
                print(f"Erreur lors de l'évaluation du modèle {filename} : {e}")

    # Affichage du meilleur modèle
    if best_model_name:
        print("\n=== Meilleur modèle trouvé ===")
        print(f"Modèle              : {best_model_name}")
        print(f" - Total loss       : {best_results[0]:.4f}")
        print(f" - Policy loss      : {best_results[1]:.4f}")
        print(f" - Value loss       : {best_results[2]:.4f}")
        print(f" - Policy accuracy  : {best_results[3]:.4f}")
        print(f" - Value accuracy   : {best_results[4]:.4f}")
        print("\n=== Summary du meilleur modèle ===")
        best_model.summary()
    else:
        print("\nAucun modèle évalué.")