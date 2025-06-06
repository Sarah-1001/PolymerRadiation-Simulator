# Auto detect text files and perform LF normalization
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigvals
import warnings
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from scipy.optimize import curve_fit

def linear_func(x, a, b):
    return a * x + b

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Handle CuPy import for CUDA support
try:
    import cupy as cp
    use_cuda = cp.is_available()
except ImportError:
    cp = None
    use_cuda = False

warnings.filterwarnings("ignore")

# User Inputs
time = int(input("Duration days: "))
Lchain = int(input("Length of chain: "))
Nchain = int(input("Number of chains: "))
Ntrials = int(input("Number of trials: "))
total_runs = int(input("Total runs: "))
dose_rate = 0.74 # Now a single integer
print("\n")

# Constants
n = Lchain
max_connections = n * Nchain

# Lists to store crosslinking and scission over time
crosslink_percentage_over_time = []
scission_percentage_over_time = []

# DQN Setup
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)

state_size = 1  # Example: time as the state
action_size = 4  # Example: 4 actions (adjustment factors)
policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
replay_buffer = ReplayBuffer(10000)
batch_size = 64
gamma = 0.99
epsilon = 10

def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    with torch.no_grad():
        return policy_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()

def optimize_model():
    if len(replay_buffer) < batch_size:
        return
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target_q_values = rewards + gamma * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main simulation loops
all_average_eigenvalues_list = []
time_vs_eigenvalue = []

for run in range(total_runs):
    print(f"\nRun {run+1} of {total_runs}")
    
    for trial in range(Ntrials):
        print(f"\nTrial {trial+1} of {Ntrials}, Run {run+1}")
        time_vs_eigenvalue = []  # <-- Reset here!

        # Calculate crosslinking and scission for this trial
        aXL_t = (7.324**-4 * time - 1.034**-3)
        bXL_t = (-5.631**-5 * time + 1.015)
        crosslink_val = -aXL_t * (dose_rate ** bXL_t)
        crosslink_percentage_over_time.append(crosslink_val)
        print(f"Crosslinking value: {crosslink_val}")

        aSC_t = ((3.385**-4 * (time ** 2)) + (3.152**-2 * time) - (4.905**-1))
        bSC_t = ((1.575**-4 * time) + 5.168**-1)
        scission_val =(aSC_t * (dose_rate ** bSC_t))
        scission_percentage_over_time.append(scission_val)
        print(f"Scission value: {scission_val}")


        initial_concentration = 0
        max_concentration =n*Nchain

        
        scission_percentage = scission_val 
        crosslink_percentage = crosslink_val 
        print(f"Scission percentage: {scission_percentage}")
        print(f"Crosslink percentage: {crosslink_percentage}")


        NXL = int((crosslink_percentage/10) * (n * Nchain))
        NSC = int((scission_percentage/10) * (n * Nchain))
        NXL = min(NXL, max_connections)
        NSC = min(NSC, max_connections)

        # Initialize graph
        G = nx.Graph()
        for chain_num in range(Nchain):
            start_node = chain_num * Lchain + 1
            chain_nodes = [str(i + start_node) for i in range(Lchain)]
            G.add_nodes_from(chain_nodes)
            G.add_edges_from((chain_nodes[i], chain_nodes[i + 1]) for i in range(len(chain_nodes) - 1))
        
        # Select action and adjust parameters
        state = [time]
        action = select_action(state)
        adjustment_factor = 1 + (action - 1.5) / 10
        # Optionally adjust NXL/NSC with DQN action if desired

        # Crosslinking (XL)
        print("============= Crosslinking ==============")
        for _ in range(NXL):
            l = np.random.randint(1, n * Nchain)
            c = np.random.randint(1, n * Nchain)
            while abs(l - c) < 1.1 or abs(l - c) >= (Nchain * Lchain) or l == c:
                c = np.random.randint(1, n * Nchain)
                l = np.random.randint(1, n * Nchain)
            G.add_edge(str(l), str(c))
            print(f"Crosslink added: {l} to {c}")

        # Scission (SC)
        print("--------------- Scission ----------------")
        max_attempts = 10000
        for _ in range(NSC):
            attempts = 0
            while attempts < max_attempts:
                edges1 = np.random.randint(1, n * Nchain + 1)
                edges2 = np.random.randint(1, n * Nchain + 1)
                if abs(edges1 - edges2) == 1 and G.has_edge(str(edges1), str(edges2)):
                    G.remove_edge(str(edges1), str(edges2))
                    print(f"Edge removed: {edges1} to {edges2}")
                    break
                attempts += 1
            if attempts == max_attempts:
                print("Warning: Unable to find a valid edge for scission")

        plt.figure(figsize=(10, 10))
        nx.draw(G, with_labels=True, node_size=50, node_color='blue', font_size=0.5)
        plt.title(f"Node Graph (Trial {trial+1}, Run {run+1})", fontsize=20)
        plt.savefig(f"C:\\Users\\sarah\\OneDrive - Washington State University (email.wsu.edu)\\Workproject polyethalene modal\\relation of xl to fxl\\node_graph_{trial+1}_{run+1}.png", dpi=100)
        plt.close()

        # Compute Laplacian matrix and eigenvalues
        L = nx.laplacian_matrix(G).todense()

        if G.number_of_nodes() == 0:
            print("Graph is empty, skipping Laplacian computation.")
            second_smallest = 0
        else:
            try:
                if use_cuda:
                    L_gpu = cp.asarray(L)
                    eigenvalues = cp.linalg.eigvalsh(L_gpu).get()
                else:
                    eigenvalues = np.real(eigvals(L))
            except Exception as e:
                print("CUDA/CPU error occurred:", e)
                eigenvalues = np.real(eigvals(L))  # Fallback to CPU

            sorted_eigenvalues = np.sort(eigenvalues)
            second_smallest = sorted_eigenvalues[1] if len(sorted_eigenvalues) > 1 else 0
            time_vs_eigenvalue.append((trial+1, second_smallest))
        
        # Save Laplacian and adjacency matrices
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_identifier = f"trial_{trial+1}_run_{run+1}"
        L = nx.laplacian_matrix(G).todense()
        plt.figure(figsize=(10, 10))
        sns.heatmap(L, annot=False, cmap='Blues')
        plt.title(f"Laplacian Matrix (Trial {trial+1}, Run {run+1})", fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig(f"C:\\Users\\sarah\\OneDrive - Washington State University (email.wsu.edu)\\Workproject polyethalene modal\\relation of xl to fxl\\laplacian_matrix_{timestamp}_{trial_identifier}.png", dpi=100)
        plt.close()

        A = nx.adjacency_matrix(G).todense()
        plt.figure(figsize=(10, 10))
        sns.heatmap(A, annot=False, cmap='Reds')
        plt.title(f"Adjacency Matrix (Trial {trial+1}, Run {run+1})", fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig(f"C:\\Users\\sarah\\OneDrive - Washington State University (email.wsu.edu)\\Workproject polyethalene modal\\relation of xl to fxl\\adjacency_matrix_{timestamp}_{trial_identifier}.png", dpi=100)
        plt.close()

        # Collect data up to current trial
        current_times = np.arange(1, trial + 2)  # +2 because trial starts at 0
        crosslink_so_far = crosslink_percentage_over_time[:trial + 1]
        scission_so_far = scission_percentage_over_time[:trial + 1]

        # Linear fit for crosslinking
        try:
            popt_lin, _ = curve_fit(linear_func, current_times, crosslink_so_far)
            crosslink_pred = linear_func(current_times, *popt_lin)
            mse_crosslink = np.mean((crosslink_so_far - crosslink_pred) ** 2)
        except Exception:
            mse_crosslink = 0  # fallback if fit fails

        # Exponential fit for scission
        try:
            popt_exp, _ = curve_fit(exp_func, current_times, scission_so_far, maxfev=10000)
            scission_pred = exp_func(current_times, *popt_exp)
            mse_scission = np.mean((scission_so_far - scission_pred) ** 2)
        except Exception:
            mse_scission = 0  # fallback if fit fails

        # Reward: negative MSE (the lower, the better)
        reward = -mse_crosslink - mse_scission

        next_state = [trial + 2]
        done = trial == Ntrials - 1
        replay_buffer.push(state, action, reward, next_state, done)
        optimize_model()

    # Update target network
    all_average_eigenvalues_list.append(list(time_vs_eigenvalue))
    target_net.load_state_dict(policy_net.state_dict())
    time+=1

#Sort by trial/time before plotting
time_vs_eigenvalue.sort(key=lambda x: x[0])  # Sort by trial/time
times, eigenvalues = zip(*time_vs_eigenvalue)

# Plot time vs. crosslinking/scission
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(crosslink_percentage_over_time)+1), crosslink_percentage_over_time, label='Crosslinking', color='blue', linewidth=4)
plt.plot(range(1, len(scission_percentage_over_time)+1), scission_percentage_over_time, label='Scission', color='red', linewidth=4)
plt.xlabel('Time (days)', fontsize=15)
plt.ylabel('Percentage', fontsize=15)
plt.title('Crosslinking and Scission vs. Time', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend()
plt.grid(True)
plt.savefig(f"C:\\Users\\sarah\\OneDrive - Washington State University (email.wsu.edu)\\Workproject polyethalene modal\\relation of xl to fxl\\crosslinking_and_scission_vs_time_{timestamp}_{trial_identifier}.png", dpi=100)
plt.close()

# Plot time vs. second smallest eigenvalue
window_size = 5  # You can adjust this window size for smoothing
eigenvalues = np.array(eigenvalues)

if len(eigenvalues) >= window_size:
    running_avg = np.convolve(eigenvalues, np.ones(window_size)/window_size, mode='valid')
    avg_times = np.array(times)[window_size-1:]  # Align times with running average

    plt.figure(figsize=(10, 6))
    plt.plot(avg_times, running_avg, label=f'Running Avg (window={window_size})', color='orange', linewidth=4)
    plt.xlabel('Time (days)', fontsize=15)
    plt.ylabel('Second Smallest Eigenvalue (Running Avg)', fontsize=15)
    plt.title('Time vs. Running Average of Second Smallest Eigenvalue', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"C:\\Users\\sarah\\OneDrive - Washington State University (email.wsu.edu)\\Workproject polyethalene modal\\relation of xl to fxl\\plot_of_running_avg_eigenvalue_vs_time_{timestamp}_{trial_identifier}.png", dpi=100)
    plt.close()
else:
    print(f"Not enough data points ({len(eigenvalues)}) for running average with window size {window_size}.")
* text=auto
