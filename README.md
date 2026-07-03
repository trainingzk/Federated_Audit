# Rational Auditing Simulation for Verifiable Vector Aggregation

This repository contains the Python implementation used to generate the experimental results for the paper on **rational auditing for verifiable vector aggregation**.

The implementation evaluates the proposed auditing protocols using Monte Carlo simulation and reproduces all figures reported in the paper.

## Overview

The simulator implements two auditing protocols:

- **Protocol P1** – each client audits a single randomly selected coordinate.
- **Protocol P2** – each client audits multiple randomly selected coordinates and averages the resulting Brier rewards.

The implementation validates the theoretical analysis by comparing empirical Monte Carlo results with the analytical predictions.

## Generated Figures

Running the program produces the following publication-quality PDF figures.

| Figure | Description |
|--------|-------------|
| `fig1.pdf` | Expected audit gap versus \(\|\delta\|^2\) for two representative numbers of clients, comparing empirical and theoretical results. |
| `fig2.pdf` | Variance of the audit score as a function of the number of audited coordinates \(k\). |
| `fig3.pdf` | Distribution of audit scores for Protocols P1 and P2 under the same dishonest aggregation. |

All figures are generated as vector PDF files suitable for inclusion in LaTeX manuscripts.

## Mathematical Model

The simulator follows the model described in the paper.

- Client vectors are generated independently.
- Each coordinate is sampled uniformly from

\[
[-1,1].
\]

Unless otherwise stated, the experiments use

- \(N=M=1\)
- vector dimension \(d=128\)

A dishonest server reports

\[
\mathbf y=\mathbf y^\ast+\boldsymbol\delta,
\]

where the attack vector is

\[
\boldsymbol\delta=(a,a,\ldots,a).
\]

Audit scores are computed using the Brier scoring rule.

## Requirements

The program requires Python 3 and the following packages:

```bash
pip install numpy matplotlib scipy
```

## Usage

Simply execute

```bash
python simulation.py
```

The program generates

```
fig1.pdf
fig2.pdf
fig3.pdf
```

in the current working directory.

## Reproducibility

The implementation uses

```python
np.random.seed(42)
```

to ensure reproducible Monte Carlo experiments.

}
```

## License

This repository is released under the MIT License.
