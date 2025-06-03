
# AlphaZero-lite pour le jeu de Go (19x19) - Version complète en .py
# Contient : modèle Keras, classe GoGame, moteur MCTS, AlphaZeroLite, entraînement

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import random
from collections import defaultdict
from tqdm import tqdm

# === PARAMÈTRES DU MODÈLE ===
l2_reg = 0.0001
filters = 16
trunk = 16
block_iteration = 5

# === BLOC SQUEEZE & EXCITATION ===
def se_block(input_tensor, filters, ratio=16):
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, use_bias=False)(se)
    se = layers.LeakyReLU()(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])

# === BLOC MOBILE-LIKE ===
def bottleneck_block(x, expand=filters, squeeze=trunk, l2_reg=l2_reg):
    m = layers.Conv2D(expand, (1,1), kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = layers.LeakyReLU()(m)
    m = layers.DepthwiseConv2D((3,3), padding='same', kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = layers.LeakyReLU()(m)
    m = layers.Conv2D(squeeze, (1,1), kernel_regularizer=regularizers.l2(l2_reg), use_bias=False)(m)
    m = layers.BatchNormalization()(m)
    m = se_block(m, squeeze)
    return layers.Add()([m, x])

# === CRÉATION DU MODÈLE ===
def get_model():
    input = keras.Input(shape=(19, 19, 31), name='board')
    x = layers.Conv2D(trunk, 1, padding='same', kernel_regularizer=regularizers.l2(l2_reg))(input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    for i in range(block_iteration):
        x = bottleneck_block(x, filters+32, trunk)
    for i in range(block_iteration):
        x = bottleneck_block(x, filters+64, trunk)
    for i in range(block_iteration):
        x = bottleneck_block(x, filters+96, trunk)
    for i in range(block_iteration):
        x = bottleneck_block(x, filters+128, trunk)
    # Policy head
    policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                kernel_regularizer=regularizers.l2(l2_reg))(x)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Activation('softmax', name='policy')(policy_head)
    # Value head
    value_head = layers.GlobalAveragePooling2D()(x)
    value_head = layers.Dense(50, kernel_regularizer=regularizers.l2(l2_reg))(value_head)
    value_head = layers.LeakyReLU()(value_head)
    value_head = layers.Dropout(0.3)(value_head)
    value_head = layers.Dense(1, activation='sigmoid', name='value',
                              kernel_regularizer=regularizers.l2(l2_reg))(value_head)
    model = keras.Model(inputs=input, outputs=[policy_head, value_head])
    return model

# === CLASSE GOGAME (version simplifiée fidèle) ===
class GoGame:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.history = []
        self.passes = 0
        self.current_player = 1
        self.previous_board = None

    def copy(self):
        new_game = GoGame(self.size)
        new_game.board = self.board.copy()
        new_game.history = [b.copy() for b in self.history]
        new_game.passes = self.passes
        new_game.current_player = self.current_player
        new_game.previous_board = self.previous_board.copy() if self.previous_board is not None else None
        return new_game

    def encode_input(self):
        planes = []

        # 8 coups précédents (16 plans)
        for i in range(8):
            if len(self.history) > i:
                board = self.history[-(i + 1)]
            else:
                board = np.zeros((self.size, self.size), dtype=np.int8)

            planes.append((board == 1).astype(np.float32))   # pierres noires
            planes.append((board == -1).astype(np.float32))  # pierres blanches

        # 2 plans = état actuel
        planes.append((self.board == 1).astype(np.float32))
        planes.append((self.board == -1).astype(np.float32))

        # 12 plans vides (réservés à l’avenir)
        for _ in range(12):
            planes.append(np.zeros((self.size, self.size), dtype=np.float32))

        # 1 plan : joueur courant
        current_player_plane = np.full((self.size, self.size), self.current_player, dtype=np.float32)
        planes.append(current_player_plane)

        return np.stack(planes, axis=-1)  # → shape: (19, 19, 31)

    def get_legal_moves(self):
        legal_moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == 0:
                    legal_moves.append((x, y))
        legal_moves.append("pass")
        return legal_moves

    def play(self, move):
        new_game = self.copy()
        if move == "pass":
            new_game.passes += 1
            new_game.current_player *= -1
            new_game.history.append(new_game.board.copy())  # Ajout historique
            return new_game

        x, y = move
        if new_game.board[x, y] != 0:
            raise ValueError("Illegal move")
        new_game.previous_board = new_game.board.copy()
        new_game.board[x, y] = new_game.current_player
        new_game.capture(x, y)
        new_game.passes = 0
        new_game.current_player *= -1
        new_game.history.append(new_game.board.copy())  # Ajout historique
        return new_game

    def capture(self, x, y):
        opponent = -self.current_player
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for dx, dy in directions:
            group = []
            liberties = []
            self._explore(x + dx, y + dy, opponent, group, liberties)
            if group and not liberties:
                for gx, gy in group:
                    self.board[gx, gy] = 0

    def _explore(self, x, y, player, group, liberties, visited=None):
        if visited is None:
            visited = set()
        if not (0 <= x < self.size and 0 <= y < self.size):
            return
        if (x, y) in visited:
            return
        visited.add((x, y))
        if self.board[x, y] == 0:
            liberties.append((x, y))
        elif self.board[x, y] == player:
            group.append((x, y))
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                self._explore(x + dx, y + dy, player, group, liberties, visited)

    def is_game_over(self):
        return self.passes >= 2

    def get_winner(self):
        return 1  # simplification : Noir gagne toujours (à remplacer par vrai score)

    def get_winner_value(self):
        return 1.0 if self.get_winner() == 1 else -1.0

    @staticmethod
    def decode_policy(policy_logits):
        size = 19
        policy = {}
        flat = policy_logits.flatten()
        for i in range(size * size):
            x, y = divmod(i, size)
            policy[(x, y)] = float(flat[i])
        if len(flat) > size * size:
            policy["pass"] = float(flat[-1])
        return policy

    @staticmethod
    def visit_counts_to_policy(visits):
        total = sum(visits.values())
        size = 19
        policy = np.zeros((size * size + 1,), dtype=np.float32)
        for move, count in visits.items():
            if move == "pass":
                policy[-1] = count / total
            else:
                x, y = move
                policy[x * size + y] = count / total
        return policy

# === MCTS + ALPHAZERO LITE ===
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = {}
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy_probs):
        legal_moves = self.state.get_legal_moves()
        for move in legal_moves:
            if move not in self.children:
                child_state = self.state.play(move)
                node = MCTSNode(child_state, parent=self, move=move)
                node.P = policy_probs.get(move, 1e-8)
                self.children[move] = node

    def is_terminal(self):
        return self.state.is_game_over()

    def select_child(self, c_puct=1.0):
        total_visits = sum(child.N for child in self.children.values())
        best_score = -float('inf')
        best_move = None
        for move, child in self.children.items():
            U = c_puct * child.P * np.sqrt(total_visits) / (1 + child.N)
            score = child.Q + U
            if score > best_score:
                best_score = score
                best_move = move
        return self.children[best_move]

    def backpropagate(self, value):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backpropagate(-value)

class AlphaZeroLite:
    def __init__(self, model, game_class, num_simulations=50):
        self.model = model
        self.num_simulations = num_simulations
        self.game_class = game_class

    def model_predict(self, state):
        board_tensor = state.encode_input()
        policy_logits, value = self.model.predict(board_tensor[None, ...], verbose=0)
        policy = self.game_class.decode_policy(policy_logits[0])
        return policy, value[0][0]

    def search(self, root):
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            while node.is_expanded() and not node.is_terminal():
                node = node.select_child()
                path.append(node)
            if not node.is_terminal():
                policy, value = self.model_predict(node.state)
                node.expand(policy)
            else:
                value = node.state.get_winner_value()
            for n in reversed(path):
                n.backpropagate(value)

    def select_move(self, state):
        root = MCTSNode(state)
        policy, _ = self.model_predict(state)
        root.expand(policy)
        self.search(root)
        visits = {move: child.N for move, child in root.children.items()}
        move = max(visits, key=visits.get)
        return move, visits

    def self_play_game(self):
        state = self.game_class()
        game_data = []
        move_count = 0
        print("\n🔄 Démarrage d'une nouvelle partie de self-play...")
        with tqdm(total=300, desc="Partie en cours", leave=False) as pbar:
            while not state.is_game_over():
                move, visit_counts = self.select_move(state)
                pbar.update(1)
                pbar.set_postfix_str(f"Coup #{move_count + 1}: {move}")
                policy_target = self.game_class.visit_counts_to_policy(visit_counts)
                board_tensor = state.encode_input()
                game_data.append((board_tensor, policy_target, state.current_player))
                state = state.play(move)
                move_count += 1
        winner = state.get_winner()
        print(f"✅ Partie terminée. Joueur gagnant: {'Noir' if winner == 1 else 'Blanc'}")
        winner_value = state.get_winner_value()
        data = []
        for board_tensor, policy_target, player in game_data:
            value = winner_value if player == state.get_winner() else -winner_value
            data.append((board_tensor, policy_target, value))
        return data

    def train_from_selfplay(self, n_games=3, batch_size=64, epochs=5):
        all_data = []
        for i in range(n_games):
            print(f"\n🎮 Self-play {i+1}/{n_games}")
            game_data = self.self_play_game()
            all_data.extend(game_data)
        print(f"\n🧠 Entraînement sur {len(all_data)} positions...")
        X = np.array([d[0] for d in all_data])
        y_policy = np.array([d[1] for d in all_data])
        y_value = np.array([[d[2]] for d in all_data])
        history = self.model.fit(X, {'policy': y_policy, 'value': y_value},
                                 batch_size=batch_size, epochs=epochs, verbose=1)
        print("✅ Entraînement terminé. Dernière loss:", history.history['loss'][-1])
