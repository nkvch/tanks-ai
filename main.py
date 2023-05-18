import torch
import torch.nn as nn
import torch.optim as optim

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
board_size = 10
max_number_of_enemies = 1
max_number_of_allies = 2
max_number_of_tanks = max_number_of_enemies + max_number_of_allies + 1
max_number_of_bullets = max_number_of_enemies + max_number_of_allies + 1
tank_representation_size = 3
bullet_representation_size = 3
board_representation_size = board_size * board_size
input_size = board_representation_size + max_number_of_tanks * tank_representation_size + max_number_of_bullets * bullet_representation_size
output_size = 5  # Number of actions

reward_map = {
    "kill_enemy": 100,
    "kill_ally": -100,
    "align_shot": 10,
    "move_closer": 1,
    "move_away": -1,
    "stay": -0.1,
    "die": -100,
}

dqn = DQN(input_size, output_size)

optimizer = optim.Adam(dqn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

board = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 = empty
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1],  # 1 = wall
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 0, 1],
]

# tank_state = [x, y, direction]
# direction: 0 = up, 1 = right, 2 = down, 3 = left
my_state = [44, 7, 2]
allies_state = [[58, 23, 0]]
enemies_state = [[20, 60, 1]]
bullets_state = []

def get_full_allies_state(allies_state):
    # use 0 to pad the state
    full_allies_state = [[0, 0, 0]] * max_number_of_allies
    for i in range(len(allies_state)):
        full_allies_state[i] = allies_state[i]
    return full_allies_state

def get_full_enemies_state(enemies_state):
    full_enemies_state = [[0, 0, 0]] * max_number_of_enemies
    for i in range(len(enemies_state)):
        full_enemies_state[i] = enemies_state[i]
    return full_enemies_state

def get_full_bullets_state(bullets_state):
    full_bullets_state = [[0, 0, 0]] * max_number_of_bullets
    for i in range(len(bullets_state)):
        full_bullets_state[i] = bullets_state[i]
    return full_bullets_state




# Convert the game state to a PyTorch tensor
def preprocess_state(state):
    return torch.tensor(state, dtype=torch.float32)

# Example usage
state = [1, 0, 1, 0, 0.5, 0.6, 3, 0.2, 0.8, 4]  # Example game state
processed_state = preprocess_state(state)

# Pass the state through the DQN to get the Q-values
q_values = dqn(processed_state)

# Choose the action with the highest Q-value
action = torch.argmax(q_values).item()

# Update the DQN with a batch of experiences
def update_dqn(batch):
    states, actions, rewards, next_states, dones = batch

    # Convert the batch to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute the Q-values for the current states and actions
    q_values = dqn(states)
    q_values = torch.gather(q_values, 1, actions.unsqueeze(1)).squeeze(1)

    # Compute the Q-values for the next states
    next_q_values = dqn(next_states).max(1)[0]

    # Compute the target Q-values
    target_q_values = rewards + (1 - dones) * next_q_values

    # Compute the loss
    loss = loss_fn(q_values, target_q_values)

    # Perform backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Example usage with a batch of experiences
# batch = (states, actions, rewards, next_states, dones)  # Example batch of experiences
# update_dqn(batch)
