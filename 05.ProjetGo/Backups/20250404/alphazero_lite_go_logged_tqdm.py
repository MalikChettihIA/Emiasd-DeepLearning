
# AlphaZeroLite avec logs + tqdm pour barre de progression
from tqdm import tqdm

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
