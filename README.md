# Quantum HVAC Optimization with QAOA

**Raphael Langer | Independent project | October 2026**

An exploration of the Quantum Approximate Optimization Algorithm (QAOA) for selecting HVAC equipment to satisfy a cooling load while minimizing electrical power consumption. The project develops a classical baseline, formulates a QUBO, maps it to an Ising cost Hamiltonian, and evaluates QAOA using exact statevectors, shot-based simulation, a noise model, and IBM quantum hardware.

## Problem

Three independently switched air-conditioning units must meet a **100 kW cooling demand**:

| Unit | Cooling capacity | Power consumption |
| --- | ---: | ---: |
| A | 50 kW | 10 kW |
| B | 75 kW | 17 kW |
| C | 100 kW | 25 kW |

The classical optimum is **A OFF, B OFF, C ON**, delivering 100 kW of cooling while consuming 25 kW of power.

## Approach

1. **Classical baseline:** Enumerate equipment configurations and identify the minimum-power feasible selection.
2. **QUBO formulation:** Use three equipment-selection bits and seven binary slack bits to represent excess cooling, giving a **10-bit / 10-qubit** problem. A penalty term encodes the cooling constraint.
3. **Ising mapping:** Convert binary variables to spin variables and construct the corresponding Pauli-Z and ZZ cost terms.
4. **QAOA circuit:** Prepare a uniform superposition with Hadamard gates, apply cost-phase rotations (`RZ`, `RZZ`), then mixer rotations (`RX`). Tune variational parameters using classical optimization with SciPy.
5. **Evaluation:** Compare exact statevector expectations, ideal shot-based results, noisy simulation, and an IBM quantum processor run.

## Results

The October 2026 report describes the following **one-layer QAOA** results (simulation samples vary between runs):

| Evaluation | Mean / expected QUBO cost | Probability of optimal state | Probability of cooling-feasible state |
| --- | ---: | ---: | ---: |
| Exact statevector | 2,372.575 | — | — |
| Ideal shot-based simulation (10,000 shots) | 2,433.2 | 0.25% | 84.0% |
| Noisy simulation (10,000 shots) | 3,125.4 | 0.23% | 81.4% |
| IBM quantum hardware (2,000 shots) | 3,977.1 | 0.4% | 70.4% |

The exact statevector expectation was **72.2% lower** than the uniform-state expected QUBO cost of 8,523.5. The reported optimized one-layer parameters were **γ = 0.7046** and **β = −0.3453**.

The hardware experiment used **IBM Kingston** (job ID `dbf8b77pc3tc739muohg`). The circuit depth increased from **21** to **250** after transpilation, with **148 two-qubit gates**. These figures are from the accompanying report; they are not claimed as freshly reproduced benchmarks.

**Interpretation:** The experiment does **not** establish quantum advantage. The optimal configuration remains rare in the measured distribution, and the noise and transpilation overhead degrade aggregate performance. The reduction relative to the uniform distribution is not a comparison against the classical optimum.

## Files

- [`Quantum HVAC Project Report.pdf`](Quantum%20HVAC%20Project%20Report.pdf) — project write-up, mathematical formulation, circuit design, results, and limitations.
- `Quantum HVAC Project Part 1.py` through `Quantum HVAC Project Part 10.py` — sequential development scripts, from classical enumeration and quantum fundamentals to QUBO/Ising construction and QAOA experiments.
- `requirements.txt` — main Python packages used in the scripts.

**Note:** These scripts were exported from an interactive development workflow. Some earlier parts rely on variables/imports defined in prior cells or parts; they are provided as a record of development and are **not presented as ten standalone, independently executable programs**. Hardware experiments additionally require authorized IBM Quantum access and may incur usage restrictions.

## Setup

```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The code was developed interactively, so run sections in their original order with the required variables initialized. The report is the recommended starting point for reviewing the completed experiment. No IBM Quantum credentials are included in this repository.

## Limitations and future work

- Explore alternative penalty scaling to avoid disproportionately large costs for infeasible configurations.
- Reduce circuit complexity and transpilation overhead on hardware.
- Evaluate additional QAOA depths and parameter-search strategies where practical.
- Extend the model to larger and more realistic HVAC equipment scheduling problems, with clear classical baselines.

This is an exploratory research/learning project, not a production HVAC control system.
