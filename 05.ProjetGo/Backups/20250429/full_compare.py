# complete_go_game.py - Système de partie complète Go avec MCTS

import numpy as np
import tensorflow as tf
from keras.models import load_model
import time
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
import math
import gc
from collections import defaultdict


class GoBoard:
    """Plateau de Go simplifié pour jouer des parties complètes"""

    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0=vide, 1=noir, 2=blanc
        self.current_player = 1  # 1=noir commence
        self.move_history = []
        self.captured_stones = {1: 0, 2: 0}
        self.komi = 6.5  # Avantage pour blanc
        self.passes = 0
        self.game_over = False

    def copy(self):
        """Copie profonde du plateau"""
        new_board = GoBoard(self.size)
        new_board.board = self.board.copy()
        new_board.current_player = self.current_player
        new_board.move_history = self.move_history.copy()
        new_board.captured_stones = self.captured_stones.copy()
        new_board.passes = self.passes
        new_board.game_over = self.game_over
        return new_board

    def get_legal_moves(self):
        """Retourne les coups légaux (incluant pass=361)"""
        legal_moves = []

        # Pass est toujours légal
        legal_moves.append(361)

        # Tester chaque intersection
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == 0:  # Case vide
                    move = row * self.size + col
                    if self.is_legal_move(move):
                        legal_moves.append(move)

        return legal_moves

    def is_legal_move(self, move):
        """Vérifie si un coup est légal (règles simplifiées)"""
        if move == 361:  # Pass
            return True

        row, col = move // self.size, move % self.size

        # Case doit être vide
        if self.board[row, col] != 0:
            return False

        # Test suicide/capture simplifié
        # Pour l'instant, on autorise tous les coups sur cases vides
        return True

    def play_move(self, move):
        """Joue un coup et retourne le nouveau plateau"""
        new_board = self.copy()

        if move == 361:  # Pass
            new_board.passes += 1
            if new_board.passes >= 2:
                new_board.game_over = True
        else:
            row, col = move // self.size, move % self.size
            new_board.board[row, col] = new_board.current_player
            new_board.passes = 0

            # Capture simplifiée (à améliorer)
            captured = new_board._remove_captured_stones(3 - new_board.current_player)
            new_board.captured_stones[new_board.current_player] += captured

        new_board.move_history.append(move)
        new_board.current_player = 3 - new_board.current_player  # Switch 1<->2

        return new_board

    def _remove_captured_stones(self, color):
        """Supprime les pierres capturées (version simplifiée)"""
        # Version très simplifiée - dans un vrai jeu, il faut gérer les libertés
        captured = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        for row in range(self.size):
            for col in range(self.size):
                if self.board[row, col] == color:
                    liberties = 0
                    for dr, dc in directions:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if self.board[nr, nc] == 0:
                                liberties += 1

                    # Si pas de libertés, capture (très simplifié)
                    if liberties == 0:
                        self.board[row, col] = 0
                        captured += 1

        return captured

    def get_winner(self):
        """Détermine le gagnant (version simplifiée)"""
        if not self.game_over:
            return None

        # Comptage simplifié: territoire = pierres + captures
        black_score = np.sum(self.board == 1) + self.captured_stones[1]
        white_score = np.sum(self.board == 2) + self.captured_stones[2] + self.komi

        if black_score > white_score:
            return 1  # Noir gagne
        else:
            return 2  # Blanc gagne

    def to_neural_input(self):
        """Convertit le plateau en input pour le réseau (simplifié)"""
        # Input basique: 3 canaux (noir, blanc, vide) + canaux additionnels
        input_planes = np.zeros((19, 19, 31), dtype=np.float32)

        # Plan 0: pierres noires
        input_planes[:, :, 0] = (self.board == 1).astype(np.float32)

        # Plan 1: pierres blanches
        input_planes[:, :, 1] = (self.board == 2).astype(np.float32)

        # Plan 2: joueur à jouer
        input_planes[:, :, 2] = float(self.current_player - 1)

        # Plans 3-30: historique et autres features (simplifiés)
        # Dans un vrai système, ajouter: libertés, échelles, historique, etc.

        return input_planes


class MCTSNode:
    """Nœud pour l'arbre MCTS"""

    def __init__(self, board_state, parent=None, move=None, prior=0.0):
        self.board_state = board_state
        self.parent = parent
        self.move = move  # Coup qui a mené à ce nœud
        self.prior = prior  # Probabilité a priori du réseau

        self.children = {}  # {move: MCTSNode}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self):
        """Valeur moyenne du nœud"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(self, c_puct=1.0):
        """Score UCT pour sélection"""
        if self.visit_count == 0:
            return float('inf')

        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.value() + exploration

    def select_child(self, c_puct=1.0):
        """Sélectionne le meilleur enfant selon UCT"""
        return max(self.children.values(), key=lambda child: child.uct_score(c_puct))

    def expand(self, policy_probs):
        """Expanse le nœud avec les probabilités du réseau"""
        legal_moves = self.board_state.get_legal_moves()

        for move in legal_moves:
            prob = policy_probs[move] if move < len(policy_probs) else 0.0
            new_board = self.board_state.play_move(move)
            self.children[move] = MCTSNode(new_board, parent=self, move=move, prior=prob)

        self.is_expanded = True

    def backup(self, value):
        """Remonte la valeur dans l'arbre"""
        self.visit_count += 1
        self.value_sum += value

        if self.parent:
            # Inverse la valeur pour l'adversaire
            self.parent.backup(-value)


class GoPlayer:
    """Joueur de Go utilisant un modèle + MCTS"""

    def __init__(self, model_path, name="Player", mcts_simulations=100):
        self.model_path = model_path
        self.name = name
        self.mcts_simulations = mcts_simulations
        self.model = None
        self.load_model()

    def load_model(self):
        """Charge le modèle"""
        try:
            self.model = load_model(self.model_path, compile=False)
            print(f"✅ Modèle {self.name} chargé: {self.model_path}")
        except Exception as e:
            print(f"❌ Erreur chargement {self.name}: {e}")

    def predict(self, board_state):
        """Prédiction du réseau pour un état de plateau"""
        if self.model is None:
            # Fallback: prédictions aléatoires MAIS favoriser les vrais coups
            policy = np.random.dirichlet([1.0] * 361 + [0.01])  # Pass très improbable
            value = random.uniform(-1, 1)
            return policy, value

        try:
            neural_input = board_state.to_neural_input()
            input_batch = np.expand_dims(neural_input, axis=0)

            pred = self.model.predict(input_batch, verbose=0)
            policy_logits = pred[0][0]  # Shape: (361,)
            value = pred[1][0][0]  # Shape: (1,)

            # 🔧 CORRECTION 1: Appliquer softmax correctement
            policy_probs = np.exp(policy_logits) / np.sum(np.exp(policy_logits))

            # 🔧 CORRECTION 2: Ajouter pass avec probabilité très faible
            policy_with_pass = np.append(policy_probs * 0.99, 0.01)  # Pass = 1%

            # 🔧 CORRECTION 3: Masquer les coups illégaux
            legal_moves = board_state.get_legal_moves()
            masked_policy = np.zeros(362)

            for move in legal_moves:
                if move < 361:  # Coup normal
                    masked_policy[move] = policy_with_pass[move]
                else:  # Pass
                    masked_policy[361] = policy_with_pass[361]

            # Renormaliser seulement les coups légaux
            if np.sum(masked_policy) > 0:
                masked_policy = masked_policy / np.sum(masked_policy)
            else:
                # Fallback: uniformément sur coups légaux
                for move in legal_moves:
                    masked_policy[move] = 1.0 / len(legal_moves)

            return masked_policy, value

        except Exception as e:
            print(f"⚠️  Erreur prédiction {self.name}: {e}")
            # Fallback intelligent
            legal_moves = board_state.get_legal_moves()
            policy = np.zeros(362)

            # Distribuer uniformément sur coups légaux (sans pass si possible)
            non_pass_moves = [m for m in legal_moves if m != 361]
            if non_pass_moves:
                for move in non_pass_moves:
                    policy[move] = 0.95 / len(non_pass_moves)
                policy[361] = 0.05  # 5% pour pass
            else:
                policy[361] = 1.0  # Seulement pass possible

            value = random.uniform(-1, 1)
            return policy, value

    def mcts_search(self, root_board):
        """Recherche MCTS pour choisir le meilleur coup"""
        root = MCTSNode(root_board)

        # 🔧 CORRECTION: Expansion immédiate du root
        policy_probs, _ = self.predict(root_board)
        root.expand(policy_probs)

        for simulation in range(self.mcts_simulations):
            # 1. Sélection
            node = root
            path = [node]

            while node.is_expanded and not node.board_state.game_over and node.children:
                node = node.select_child()
                path.append(node)

            # 2. Expansion et évaluation
            if not node.board_state.game_over:
                policy_probs, value = self.predict(node.board_state)
                if not node.is_expanded:
                    node.expand(policy_probs)

                # 🔧 CORRECTION: Prendre en compte le joueur actuel pour la value
                if node.board_state.current_player != root_board.current_player:
                    value = -value  # Inverser pour l'adversaire

            else:
                # Jeu terminé
                winner = node.board_state.get_winner()
                if winner == root_board.current_player:
                    value = 1.0
                elif winner == (3 - root_board.current_player):
                    value = -1.0
                else:
                    value = 0.0  # Égalité

            # 3. Remontée
            for path_node in reversed(path):
                path_node.backup(value)
                value = -value  # Inverser pour chaque niveau

        # 🔧 CORRECTION: Debug et choix plus intelligent
        if not root.children:
            print(f"⚠️  {self.name}: Pas d'enfants, passe forcé")
            return 361

        # Afficher les top moves pour debug
        if self.name in ["Me", "Competitor"]:  # Debug pour les modèles principaux
            sorted_moves = sorted(root.children.items(),
                                  key=lambda x: x[1].visit_count, reverse=True)
            top_3 = sorted_moves[:3]

            print(f"🤔 {self.name} - Top moves:")
            for move, node in top_3:
                if move == 361:
                    print(f"   Pass: {node.visit_count} visites, valeur={node.value():.3f}")
                else:
                    row, col = move // 19, move % 19
                    print(f"   {chr(ord('A') + col)}{19 - row}: {node.visit_count} visites, valeur={node.value():.3f}")

        # Choisir le coup le plus visité
        best_move = max(root.children.keys(),
                        key=lambda move: root.children[move].visit_count)

        return best_move

    def choose_move(self, board_state):
        """Choisit un coup pour l'état donné"""
        if board_state.game_over:
            return 361  # Pass

        return self.mcts_search(board_state)


class GoGameManager:
    """Gestionnaire de partie complète entre deux joueurs"""

    def __init__(self, player1, player2, verbose=True):
        self.player1 = player1  # Joue noir
        self.player2 = player2  # Joue blanc
        self.verbose = verbose

    def play_game(self, max_moves=400):
        """Joue une partie complète"""
        board = GoBoard()
        move_count = 0

        if self.verbose:
            print(f"🎮 NOUVELLE PARTIE: {self.player1.name} (Noir) vs {self.player2.name} (Blanc)")
            print("=" * 60)

        while not board.game_over and move_count < max_moves:
            current_player = self.player1 if board.current_player == 1 else self.player2

            if self.verbose:
                print(
                    f"\n🔄 Coup {move_count + 1} - {current_player.name} ({['', 'Noir', 'Blanc'][board.current_player]})")

            # Chronométrer le coup
            start_time = time.time()
            move = current_player.choose_move(board)
            move_time = time.time() - start_time

            # Jouer le coup
            board = board.play_move(move)
            move_count += 1

            if self.verbose:
                if move == 361:
                    print(f"   ✋ Pass ({move_time:.2f}s)")
                else:
                    row, col = move // 19, move % 19
                    print(f"   🎯 {chr(ord('A') + col)}{19 - row} ({move_time:.2f}s)")

                print(
                    f"   📊 Pierres: Noir={np.sum(board.board == 1)}, Blanc={np.sum(board.board == 2)}, Passes consécutifs={board.passes}")

                # 🔧 CORRECTION: Arrêter si trop de passes consécutifs
                if board.passes >= 2:
                    print(f"   🏁 Fin de partie (2 passes consécutifs)")
                    break

            # Nettoyage mémoire périodique
            if move_count % 10 == 0:
                gc.collect()

        # Résultat
        winner = board.get_winner()
        if self.verbose:
            print("\n" + "=" * 60)
            print("🏁 PARTIE TERMINÉE")
            print(f"🏆 Vainqueur: {['', self.player1.name + ' (Noir)', self.player2.name + ' (Blanc)'][winner]}")
            print(f"📈 Nombre de coups: {move_count}")
            print("=" * 60)

        return {
            'winner': winner,
            'winner_name': ['Égalité', self.player1.name, self.player2.name][winner] if winner else 'Égalité',
            'move_count': move_count,
            'final_board': board,
            'black_score': np.sum(board.board == 1) + board.captured_stones[1],
            'white_score': np.sum(board.board == 2) + board.captured_stones[2] + board.komi
        }


def play_tournament(model1_path, model2_path, num_games=10,
                    model1_name="Model1", model2_name="Model2",
                    mcts_simulations=50):
    """Organise un tournoi entre deux modèles"""

    print("🏆 TOURNOI DE GO COMPLET")
    print("=" * 50)
    print(f"🎯 {num_games} parties avec {mcts_simulations} simulations MCTS")
    print(f"⚫ {model1_name}: {model1_path}")
    print(f"⚪ {model2_name}: {model2_path}")
    print("=" * 50)

    # Créer les joueurs
    player1 = GoPlayer(model1_path, model1_name, mcts_simulations)
    player2 = GoPlayer(model2_path, model2_name, mcts_simulations)

    # 🔧 TEST: Vérifier que les modèles ne passent pas tout de suite
    print("\n🧪 TEST RAPIDE DES MODÈLES:")
    test_board = GoBoard()

    print("Test Model1:")
    policy1, value1 = player1.predict(test_board)
    print(f"   Pass probability: {policy1[361]:.3f}")
    print(f"   Max move probability: {np.max(policy1[:361]):.3f}")

    print("Test Model2:")
    policy2, value2 = player2.predict(test_board)
    print(f"   Pass probability: {policy2[361]:.3f}")
    print(f"   Max move probability: {np.max(policy2[:361]):.3f}")

    print("=" * 50)

    # Statistiques
    wins = {model1_name: 0, model2_name: 0, 'Égalité': 0}
    games_as_black = {model1_name: 0, model2_name: 0}
    games_as_white = {model1_name: 0, model2_name: 0}
    total_moves = []
    game_times = []

    for game_num in range(num_games):
        print(f"\n🎮 PARTIE {game_num + 1}/{num_games}")

        # Alterner les couleurs
        if game_num % 2 == 0:
            black_player, white_player = player1, player2
            games_as_black[model1_name] += 1
            games_as_white[model2_name] += 1
        else:
            black_player, white_player = player2, player1
            games_as_black[model2_name] += 1
            games_as_white[model1_name] += 1

        # Jouer la partie
        manager = GoGameManager(black_player, white_player, verbose=False)

        start_time = time.time()
        result = manager.play_game()
        game_time = time.time() - start_time

        # Enregistrer résultats
        winner_name = result['winner_name']
        wins[winner_name] += 1
        total_moves.append(result['move_count'])
        game_times.append(game_time)

        print(f"   🏆 Vainqueur: {winner_name}")
        print(f"   ⏱️  Durée: {game_time:.1f}s ({result['move_count']} coups)")

        # Nettoyage mémoire
        gc.collect()

    # Résultats finaux
    print("\n" + "=" * 50)
    print("📊 RÉSULTATS FINAUX")
    print("=" * 50)

    print(f"🏆 Victoires:")
    for player, win_count in wins.items():
        winrate = (win_count / num_games) * 100
        print(f"   • {player}: {win_count}/{num_games} ({winrate:.1f}%)")

    print(f"\n⚫ Parties comme Noir:")
    for player, count in games_as_black.items():
        print(f"   • {player}: {count} parties")

    print(f"\n⚪ Parties comme Blanc:")
    for player, count in games_as_white.items():
        print(f"   • {player}: {count} parties")

    print(f"\n📈 Statistiques:")
    print(f"   • Coups moyens par partie: {np.mean(total_moves):.1f}")
    print(f"   • Temps moyen par partie: {np.mean(game_times):.1f}s")
    print(f"   • Temps total tournoi: {sum(game_times):.1f}s")

    # Déterminer le vainqueur du tournoi
    tournament_winner = max(wins.keys(), key=lambda k: wins[k] if k != 'Égalité' else -1)

    print(f"\n🥇 VAINQUEUR DU TOURNOI: {tournament_winner}")
    print("=" * 50)

    return {
        'wins': wins,
        'games_as_black': games_as_black,
        'games_as_white': games_as_white,
        'average_moves': np.mean(total_moves),
        'average_time': np.mean(game_times),
        'tournament_winner': tournament_winner
    }


if __name__ == "__main__":
    # Configuration du tournoi
    model1_path = "Test10/best_model_epoch160_val2.8053_aug8x.h5"
    model2_path = "detheve_vrel_18052025-0.4626.h5"

    # Lancer le tournoi
    results = play_tournament(
        model1_path=model1_path,
        model2_path=model2_path,
        num_games=5,  # Commencer avec peu de parties pour tester
        model1_name="Me",
        model2_name="Competitor",
        mcts_simulations=30  # Réduire pour commencer
    )

    print(f"\n🎉 Tournoi terminé! Vainqueur: {results['tournament_winner']}")