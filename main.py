import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from EnergyEnv import EnergyEnv
from matplotlib import pyplot as plt
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
    def forward(self, x):
        return self.fc(x)

def train_qlearning(env, episodes=100):
    qnet = QNetwork(2,2)  # d'abord on crée le modèle
 
    # S'il existe déjà un modèle sauvegardé, on le charge pour continuer l'entraînement
    if os.path.exists("qnet_model.pth"):
        qnet.load_state_dict(torch.load("qnet_model.pth"))
        print("Modèle existant chargé pour entraînement continu.")
    else:
        print("Nouveau modèle créé.")

    rewards = []
    optimizer = optim.Adam(qnet.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    gamma = 0.9
    epsilon = 0.1

    for episode in range(episodes):
        state = torch.FloatTensor(env.reset())
        total_reward = 0

        while not env.done:
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                action = torch.argmax(qnet(state)).item()

            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)
            target = reward + gamma * torch.max(qnet(next_state)).item()
            output = qnet(state)[action]

            loss = criterion(output, torch.tensor(target, dtype=torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        epsilon = max(0.01, epsilon * 0.995)  # Décroissance de epsilon pour mieux exploiter plus tard
        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")
    
    torch.save(qnet.state_dict(), "qnet_model.pth")
    print(f"✅ Fin de l'entraînement sur {episodes} épisodes.")  
    plt.plot(range(1, len(rewards) + 1), rewards)

    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Evolution du total reward pendant l'entraînement")
    plt.show()

if __name__ == "__main__":
    print("=== 🚗 Hybrid Energy RL Demo ===")
    env = EnergyEnv()
    train_qlearning(env)
