# Quantum Optimization for HVAC Equipment Selection

## Overview

This project investigates the use of quantum optimization methods for a simplified HVAC equipment-selection problem. Given equipment options with different capacities and costs, the objective is to select a combination that meets a specified demand while minimizing cost.

The project is a learning and research exploration of **quadratic unconstrained binary optimization (QUBO)**, its mapping to an **Ising Hamiltonian**, and the **Quantum Approximate Optimization Algorithm (QAOA)** implemented with Qiskit. It does not claim quantum advantage.

## Problem formulation

For binary equipment-selection variables \(x_i \in \{0,1\}\), the basic optimization problem is:

\[
\min \sum_i c_i x_i
\quad \text{subject to} \quad
\sum_i q_i x_i \ge D,
\]

where \(c_i\) is equipment cost, \(q_i\) is equipment capacity, and \(D\) is required capacity.

The illustrative test instance uses:

| Equipment | Capacity | Cost (arbitrary units) |
|---|---:|---:|
| A | 50 kW | 10 |
| B | 75 kW | 17 |
| C | 100 kW | 25 |

Required capacity: **100 kW**. Slack variables and a finite penalty coefficient are used to express the constrained problem as a QUBO.

## Approach

1. Encode equipment choices and the demand constraint as a QUBO.
2. Convert the QUBO objective into an Ising cost Hamiltonian.
3. Build parameterized QAOA circuits with Qiskit.
4. Optimize QAOA parameters using a classical optimizer (including COBYLA experiments).
5. Compare candidate solutions with the classical optimum and examine the effect of QAOA depth and parameter choices.

## Current status and limitations

An initial depth-one QAOA implementation has been explored. Early experiments indicate that finding high-quality solutions is challenging for the current parameterization. Results are preliminary and should not be interpreted as evidence of quantum speedup or advantage.

Planned improvements include testing deeper QAOA circuits, normalizing or rescaling the cost Hamiltonian, running multiple optimizer restarts, and benchmarking against exact classical enumeration or optimization.

## Repository contents

- **Code:** QUBO construction, QAOA circuits, and optimization experiments (see the uploaded Python scripts or notebooks).
- **Report:** Detailed derivations, methodology, experiments, and discussion (see the uploaded report PDF).
- **Figures:** Plots and experiment outputs, if included.

> Update the filenames and links in this section after uploading the actual files.

## Running the project

This project uses **Python** and **Qiskit**, with scientific Python libraries such as **NumPy** and **SciPy**. Refer to the imports in the uploaded scripts/notebooks for the exact package requirements and versions.

If a `requirements.txt` is provided, install dependencies with:

```bash
python -m pip install -r requirements.txt
```

Then open the notebook in Jupyter or run the relevant Python script. Execution steps and package versions should be verified against the uploaded code before publishing.

## Research motivation

The broader motivation is to understand how quantum algorithms can be formulated, implemented, and evaluated for engineering optimization problems, and to explore their potential applicability to more complex scheduling, control, and autonomous-system problems.
