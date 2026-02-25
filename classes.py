import numpy as np
import torch
import torch.nn as nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device.upper()}")

class Board:
    def __init__(self):
        self.raw = np.zeros((3, 3), dtype=np.int8)
        self.player = True

    def legal_moves(self):
        return np.where(0 == self.raw.ravel())[0]

    def check_win(self):
        row_sums = np.sum(self.raw, axis=1)
        col_sums = np.sum(self.raw, axis=0)

        diag_sum = np.trace(self.raw)
        anti_diag_sum = np.trace(np.fliplr(self.raw))

        all_sums = np.concatenate([row_sums, col_sums, [diag_sum, anti_diag_sum]])

        if 3 in all_sums:
            return 1
        if -3 in all_sums:
            return -1

        if not np.any(self.raw == 0):
            return 0

        return None

    def make_move(self, move):
        if self.raw.ravel()[move] != 0:
            raise ValueError("Illegal move")

        row, col = divmod(move, 3)
        self.raw[row, col] = 1 if self.player else -1
        self.player = not self.player

    def get_boards(self):
        if self.player:
            board = self.raw.ravel()
        else:
            board = -1 * self.raw.ravel()

        player_one = board == 1
        player_two = board == -1
        empty = board == 0

        return np.stack((player_one, player_two, empty)).ravel()


class TicTacToeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_stack = nn.Sequential(
            nn.Linear(27, 27*27),
            nn.ReLU(),
            nn.Linear(27*27, 27*27),
            nn.ReLU(),
            nn.Linear(27*27, 9),
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits

class Agent:
    def __init__(self, model, temp=1.0, lr=0.01):
        self.model = model.to(device)
        self.temp = temp
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def sel_move(self, board):
        legal_moves = board.legal_moves()

        state_planes = board.get_boards()
        state_tensor = torch.FloatTensor(state_planes).unsqueeze(0).to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(state_tensor).squeeze(0)

        mask = torch.full_like(logits, float("-inf"))
        mask[torch.tensor(legal_moves, device=device)] = 0
        masked_logits = logits + mask

        probs = torch.softmax(masked_logits / self.temp, dim=0)

        move = torch.multinomial(probs, num_samples=1).item()
        return move

    def train_on_game(self, game_history, outcome):
        """
        Train an agent's model based on game outcome.
        outcome: 1 if agent won, -1 if lost, 0 if draw
        If agent won: encourage all moves with cross entropy loss
        If agent lost: discourage only the last move with cross entropy loss
        """
        agent_won = (outcome == 1)

        # Get all moves for this agent
        agent_moves = [move_data for move_data in game_history if move_data['agent'] == 0]

        if not agent_moves or outcome == 0:
            return 0

        total_loss = 0
        num_trained = 0

        for i, move_data in enumerate(agent_moves):
            state = torch.FloatTensor(move_data['state']).unsqueeze(0).to(device)
            move = move_data['move']

            # Skip losing moves except the last one
            if not agent_won and i != len(agent_moves) - 1:
                continue

            # Get model prediction
            self.model.train()
            logits = self.model(state)

            # Calculate cross entropy loss
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([move], device=device))

            # Backprop and update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_trained += 1

        return total_loss / num_trained if num_trained > 0 else 0
