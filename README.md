# Rational Auditing for Verifiable Vector Aggregation in Federated Learning

This repository contains the Python implementation used to generate the experimental results for the paper on **rational auditing for verifiable vector aggregation** in federated prototype learning.

The implementation evaluates the proposed auditing protocols (P1 and P2) within a **realistic Federated Prototype Learning (FedProto)** environment on the MNIST dataset, rather than on synthetic vectors. It reproduces all figures reported in the paper.

## Overview

The code simulates a federated learning system with:

- Non‑IID data partitioning among clients using a Dirichlet distribution with concentration parameter \(\alpha = 0.5\).
- A lightweight CNN feature extractor (two convolutional layers with 32 and 64 filters, ReLU, max‑pooling, followed by two linear layers) that produces 128‑dimensional embeddings.
- FedProto aggregation (summation of prototypes) at the server.
- Auditing protocols applied verbatim to the real prototypes.

The two audit protocols are:

- **Protocol P1** – each client audits a single randomly selected coordinate.
- **Protocol P2** – each client audits multiple randomly selected coordinates and averages the resulting Brier rewards.

The implementation validates the theoretical analysis by comparing empirical results from the FedProto rounds with the analytical predictions.

## Generated Figures

Running the program produces the following publication‑quality PDF figures.

| Figure | Description |
|--------|-------------|
| `fig1.pdf` | Expected audit gap versus \(\|\delta\|^2\) for two representative numbers of clients, measured during FedProto training on MNIST. |
| `fig2.pdf` | Variance of the audit score as a function of the number of audited coordinates \(k\) under Protocol P2. |
| `fig3.pdf` | Histograms of audit scores for P1 and P2 under the same dishonest aggregation. |

All figures are generated as vector PDF files suitable for inclusion in LaTeX manuscripts.

## Mathematical Model

The simulator follows the model described in the paper, but with prototypes **obtained from actual FedProto training** on MNIST.

- Client vectors \(\mathbf{x}_i\) are the class‑0 prototypes computed by each client’s local model.
- Each prototype is bounded in \([-N, M]^d\) with \(N=M=1\).
- Vector dimension \(d = 128\).
- Dishonest aggregation is simulated by adding \(\boldsymbol\delta = (a, a, \ldots, a)\) to the true aggregate \(\mathbf{y}^*\).

Audit scores are computed using the Brier scoring rule exactly as defined in the paper.

## Requirements

The program requires Python 3 and the following packages:

```bash
pip install numpy matplotlib scipy torch torchvision

- PyTorch and torchvision are used for the FedProto simulation.
- All other packages (NumPy, Matplotlib, SciPy) are used for the audit and plotting.

## Usage

Simply execute:

```bash
python fedproto_audit.py
```

The script will:

1. Download MNIST automatically.
2. Partition the data among \(n=20\) and \(n=80\) clients using a Dirichlet(0.5) split.
3. Run multiple FedProto rounds to collect client prototypes.
4. Apply the auditing protocols (P1 and P2) to the prototypes.
5. Generate the three figures:  
   `fig1.pdf`, `fig2.pdf`, `fig3.pdf`

All random seeds are fixed for reproducibility.

## Reproducibility

The implementation uses fixed seeds for both NumPy and PyTorch to ensure reproducible results.

```python
np.random.seed(42)
torch.manual_seed(42)
```

The code is self‑contained and does not require external data files (MNIST is downloaded automatically).

## License

This repository is released under the MIT License.
