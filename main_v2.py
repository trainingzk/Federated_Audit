import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Global settings
# ============================================================
np.random.seed(42)
torch.manual_seed(42)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 6
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["grid.alpha"] = 0.3

# ============================================================
# Mathematical helpers (unchanged)
# ============================================================

def brier_reward(u, b):
    """Brier reward R(u; b) for vectorized inputs."""
    return np.where(b == 1, 2*u - u*u, 1 - u*u)

def simulate_p1(clients, y_claim, M, N, n_reps):
    """
    Simulate Protocol P1 (k=1) for multiple independent rounds.
    clients: (n, d) array
    y_claim: (d,) array
    Returns: (n_reps,) array of total audit scores T1.
    """
    n, d = clients.shape
    u_j = (y_claim + n * N) / (n * (M + N))
    p_ij = (clients + N) / (M + N)
    p_flat = p_ij.ravel()

    coords = np.random.randint(0, d, size=(n_reps, n))
    rows = np.arange(n).reshape(1, n)
    flat_idx = rows * d + coords
    p_sel = p_flat[flat_idx]
    u_sel = u_j[coords]
    b = np.random.rand(n_reps, n) < p_sel
    rewards = brier_reward(u_sel, b)
    T1 = rewards.sum(axis=1)
    return T1

def simulate_p2(clients, y_claim, k, M, N, n_reps):
    """
    Simulate Protocol P2 with k coordinates per client.
    clients: (n, d) array
    y_claim: (d,) array
    Returns: (n_reps,) array of total audit scores T_k.
    """
    n, d = clients.shape
    u_j = (y_claim + n * N) / (n * (M + N))
    p_ij = (clients + N) / (M + N)
    p_flat = p_ij.ravel()

    total_pairs = n_reps * n
    rand = np.random.rand(total_pairs, d)
    coords_flat = rand.argsort(axis=1)[:, :k]
    coords = coords_flat.reshape(n_reps, n, k)

    rows = np.arange(n).reshape(1, n, 1)
    flat_idx = rows * d + coords
    p_sel = p_flat[flat_idx]
    u_sel = u_j[coords]
    b = np.random.rand(n_reps, n, k) < p_sel
    rewards = brier_reward(u_sel, b)
    Z_i = rewards.mean(axis=2)
    Tk = Z_i.sum(axis=1)
    return Tk

# ============================================================
# FedProto components
# ============================================================

class SmallCNN(nn.Module):
    """Small CNN feature extractor for MNIST, output dimension d."""
    def __init__(self, embedding_dim=128):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, embedding_dim)  # embedding layer
        self.classifier = nn.Linear(embedding_dim, 10)  # for local training

    def forward(self, x, return_embedding=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        emb = self.fc2(x)
        if return_embedding:
            return emb
        logits = self.classifier(emb)
        return logits

def create_non_iid_partition(dataset, n_clients, alpha=0.5, seed=42):
    """Partition dataset into n_clients using Dirichlet distribution."""
    np.random.seed(seed)
    labels = np.array(dataset.targets)
    n_classes = len(dataset.classes)
    idx_per_class = [np.where(labels == c)[0] for c in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        # Dirichlet distribution over clients
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        # ensure each client gets at least one sample per class
        proportions = np.maximum(proportions, 1e-3)
        proportions /= proportions.sum()
        n_samples = len(idx_per_class[c])
        assigned = np.random.choice(n_clients, n_samples, p=proportions)
        for i, idx in enumerate(idx_per_class[c]):
            client_indices[assigned[i]].append(idx)
    # Shuffle each client's list
    for i in range(n_clients):
        np.random.shuffle(client_indices[i])
    return client_indices

def train_local_model(model, dataloader, epochs=1, lr=0.01):
    """Train a local model for one epoch (or more)."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

def compute_prototype(model, dataloader, class_id=0):
    """Compute prototype for a given class as mean of embeddings."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, labels in dataloader:
            # only consider samples of the target class
            mask = labels == class_id
            if mask.sum() == 0:
                continue
            emb = model(images, return_embedding=True)
            embeddings.append(emb[mask].cpu().numpy())
    if len(embeddings) == 0:
        # fallback: return zeros if no samples of this class
        return np.zeros(model.fc2.out_features)
    all_emb = np.concatenate(embeddings, axis=0)
    return all_emb.mean(axis=0)

# ============================================================
# Main FedProto simulation and audit data collection
# ============================================================

def run_fedproto_rounds(n_clients, d, n_rounds, batch_size=64, local_epochs=1, alpha=0.5):
    """Run FedProto training and collect client prototypes for each round."""
    # Load MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Partition data
    client_indices = create_non_iid_partition(train_dataset, n_clients, alpha=alpha)
    client_datasets = [Subset(train_dataset, indices) for indices in client_indices]
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    # Initialize global model (shared across clients)
    global_model = SmallCNN(embedding_dim=d)

    # Storage for prototypes per round: list of (n_clients, d) arrays
    all_round_prototypes = []

    for round_idx in range(n_rounds):
        # Local training and prototype computation
        client_prototypes = []
        for client_id in range(n_clients):
            # Copy global model to local
            local_model = SmallCNN(embedding_dim=d)
            local_model.load_state_dict(global_model.state_dict())
            # Train locally
            train_local_model(local_model, client_loaders[client_id], epochs=local_epochs)
            # Compute prototype for class 0
            proto = compute_prototype(local_model, client_loaders[client_id], class_id=0)
            client_prototypes.append(proto)
        client_prototypes = np.array(client_prototypes)  # shape (n_clients, d)
        all_round_prototypes.append(client_prototypes)

        # Aggregate prototypes (FedProto) - here we just do sum, but we can also average
        # In paper's notation, aggregate is sum, so we keep it.
        # For next round, we could use average of prototypes as global model's initialization?
        # But for audit we don't need to update the model. We just need the prototypes.
        # However, for realism, we might want to update the global model by averaging prototypes? Not needed for audit.
        # We'll keep the global model unchanged for simplicity.

    return all_round_prototypes

# ============================================================
# Figure generation using collected prototypes
# ============================================================

def generate_fig1(all_round_prototypes, n_values=[20, 80], d=128, M=1, N=1, n_reps=3000, n_rounds_use=20):
    """Generate Figure 1: expected audit gap vs ||delta||^2."""
    # We need to run for each n in n_values
    # But we have all_round_prototypes per n? We need separate runs for each n.
    # We'll run the whole FedProto for each n, collect prototypes.
    # For simplicity, we'll create a wrapper that runs the whole thing for each n.
    # This will be called inside the figure generation.
    # We'll handle that in main.

    pass  # will implement in main

# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print("Starting FedProto audit experiments...")
    # Parameters
    d = 128
    M = N = 1.0
    n_rounds = 20  # number of FL rounds
    local_epochs = 1
    alpha = 0.5
    batch_size = 64

    # We will run for two client counts: 20 and 80 (to keep runtime moderate)
    n_list = [20, 80]
    all_prototypes_dict = {}

    for n in n_list:
        print(f"Running FedProto for n={n} clients...")
        prototypes = run_fedproto_rounds(n, d, n_rounds, batch_size, local_epochs, alpha)
        all_prototypes_dict[n] = prototypes  # list of (n,d) arrays per round
        print(f"Collected {len(prototypes)} rounds of prototypes.")

    # Figure 1: Expected audit gap vs ||delta||^2
    print("Generating fig1.pdf ...")
    # Use the collected prototypes to compute audit gap for various delta norms
    a_vals = np.linspace(0, 1.5, 20)
    delta_sq = d * a_vals**2
    n_reps = 3000  # number of Monte Carlo repetitions per (round, delta)
    plt.figure(figsize=(6, 4.5))

    for idx, n in enumerate(n_list):
        prototypes_rounds = all_prototypes_dict[n]  # list of arrays
        # Compute true aggregate for each round (sum of prototypes)
        true_aggregates = [np.sum(protos, axis=0) for protos in prototypes_rounds]
        # Average across rounds for each delta
        gaps = []
        for a in a_vals:
            if a == 0:
                gaps.append(0.0)
                continue
            delta_vec = a * np.ones(d)
            # For each round, compute gap and average
            round_gaps = []
            for round_idx, protos in enumerate(prototypes_rounds):
                y_true = true_aggregates[round_idx]
                y_dish = y_true + delta_vec
                # Run audit simulation on this round's clients
                T1_honest = simulate_p1(protos, y_true, M, N, n_reps)
                T1_dish = simulate_p1(protos, y_dish, M, N, n_reps)
                gap = T1_honest.mean() - T1_dish.mean()
                round_gaps.append(gap)
            gaps.append(np.mean(round_gaps))
        theor_gap = delta_sq / (d * n * (M + N)**2)

        plt.plot(delta_sq, gaps, 'o-', label=f'Simulation (n={n})', color=f'C{idx}')
        plt.plot(delta_sq, theor_gap, '--', label=f'Theory (n={n})', color=f'C{idx}', linewidth=2.5)

    plt.xlabel(r'$||\delta||^2$')
    plt.ylabel('Expected audit gap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1.pdf')
    plt.close()
    print("fig1.pdf saved.")

    # Figure 2: Variance reduction vs k
    print("Generating fig2.pdf ...")
    n = 80  # choose one n
    prototypes_rounds = all_prototypes_dict[n]
    # Use the true aggregate of the first round (or average across rounds)
    # We'll compute variance for each k using many repetitions
    k_list = [1, 2, 4, 8, 16, 32, 64]
    n_reps_var = 2000
    # For each round, compute variance and average
    var_emp_list = []
    for round_idx, protos in enumerate(prototypes_rounds):
        y_true = np.sum(protos, axis=0)
        var_round = []
        for k in k_list:
            Tk = simulate_p2(protos, y_true, k, M, N, n_reps_var)
            var_round.append(np.var(Tk))
        var_emp_list.append(var_round)
    var_emp = np.mean(var_emp_list, axis=0)
    # Theoretical: scaled 1/k
    theor = (var_emp[0] / 1.0) / np.array(k_list)

    plt.figure(figsize=(6, 4.5))
    plt.plot(k_list, var_emp, 'o-', label='Empirical variance')
    plt.plot(k_list, theor, '--', label='Theoretical 1/k (scaled)')
    plt.xlabel('Number of coordinates k')
    plt.ylabel('Variance of audit score')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig2.pdf')
    plt.close()
    print("fig2.pdf saved.")

    # Figure 3: Histograms for P1 and P2 with dishonest server
    print("Generating fig3.pdf ...")
    n = 80
    prototypes_rounds = all_prototypes_dict[n]
    # Pick a round (e.g., last round)
    protos = prototypes_rounds[-1]
    y_true = np.sum(protos, axis=0)
    target_delta_sq = 64.0
    a = np.sqrt(target_delta_sq / d)
    delta_vec = a * np.ones(d)
    y_dish = y_true + delta_vec
    n_reps_hist = 5000
    T1 = simulate_p1(protos, y_dish, M, N, n_reps_hist)
    T16 = simulate_p2(protos, y_dish, 16, M, N, n_reps_hist)

    plt.figure(figsize=(6, 4.5))
    plt.hist(T1, bins=50, alpha=0.6, label='P1 (k=1)', density=True, color='blue')
    plt.hist(T16, bins=50, alpha=0.6, label='P2 (k=16)', density=True, color='red')
    plt.xlabel('Audit score T')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig3.pdf')
    plt.close()
    print("fig3.pdf saved.")

    print("All figures generated successfully using FedProto prototypes.")
