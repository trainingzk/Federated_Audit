import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time

# ============================================================
# Global settings
# ============================================================
np.random.seed(42)
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
# Mathematical helpers
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
# Figure 1: Expected audit gap vs squared deviation
# ============================================================

def generate_fig1():
    d = 128
    M = N = 1.0
    n_values = [20, 160]                     # Only two n values
    n_reps = 15000                            # Increased for smoothness

    # Generate about 20 points for a
    a_vals = np.linspace(0, 1.5, 20)
    delta_sq = d * a_vals**2

    plt.figure(figsize=(6, 4.5))

    for n in n_values:
        clients = np.random.uniform(-N, M, size=(n, d))
        y_true = clients.sum(axis=0)

        # Precompute honest scores for this n
        T1_honest = simulate_p1(clients, y_true, M, N, n_reps)
        mean_honest = T1_honest.mean()

        gaps = []
        for a in a_vals:
            if a == 0:
                gaps.append(0.0)
                continue
            delta_vec = a * np.ones(d)
            y_dish = y_true + delta_vec
            T1_dish = simulate_p1(clients, y_dish, M, N, n_reps)
            mean_dish = T1_dish.mean()
            gaps.append(mean_honest - mean_dish)

        theor_gap = delta_sq / (d * n * (M + N)**2)

        # Simulation: solid line with markers
        plt.plot(delta_sq, gaps, 'o-', label=f'Simulation (n={n})', color=f'C{n_values.index(n)}')
        # Theory: dashed line, slightly thicker
        plt.plot(delta_sq, theor_gap, '--', label=f'Theory (n={n})', 
                 color=f'C{n_values.index(n)}', linewidth=2.5)

    plt.xlabel(r'$||\delta||^2$')
    plt.ylabel('Expected audit gap')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig1.pdf')
    plt.close()

# ============================================================
# Figure 2: Variance reduction vs k (unchanged)
# ============================================================

def generate_fig2():
    n = 80
    d = 128
    M = N = 1.0
    k_list = [1, 2, 4, 8, 16, 32, 64]
    n_reps = 2000

    clients = np.random.uniform(-N, M, size=(n, d))
    y_true = clients.sum(axis=0)

    var_emp = []
    for k in k_list:
        Tk = simulate_p2(clients, y_true, k, M, N, n_reps)
        var_emp.append(np.var(Tk))

    var_emp = np.array(var_emp)
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

# ============================================================
# Figure 3: Histograms for P1 and P2 with dishonest server (unchanged)
# ============================================================

def generate_fig3():
    n = 80
    d = 128
    M = N = 1.0
    n_reps = 5000
    target_delta_sq = 64.0
    a = np.sqrt(target_delta_sq / d)

    clients = np.random.uniform(-N, M, size=(n, d))
    y_true = clients.sum(axis=0)
    delta_vec = a * np.ones(d)
    y_dish = y_true + delta_vec

    T1 = simulate_p1(clients, y_dish, M, N, n_reps)
    T16 = simulate_p2(clients, y_dish, 16, M, N, n_reps)

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

# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    print("Generating fig1.pdf ...")
    generate_fig1()
    print("Generating fig2.pdf ...")
    generate_fig2()
    print("Generating fig3.pdf ...")
    generate_fig3()
    print("All figures generated successfully.")