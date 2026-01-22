# BiLSTM-Based-Framework-for-Constitutive-Model-Parameter-Identification

[![ETH Zurich](https://img.shields.io/badge/ETH-Zurich-blue)](https://ethz.ch)
[![Empa](https://img.shields.io/badge/Empa-Collaboration-green)](https://www.empa.ch)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org)

A deep learning framework for **one-shot identification** of Rubin–Bodner constitutive model parameters from stress-strain curves. Replaces hours of iterative optimization with millisecond-scale inference.

---

## Key Features

- **BiLSTM Architecture** — Processes time-dependent stress-strain histories with memory
- **10 Material Parameters** — Predicts all RB model parameters in a single forward pass
- **Multiple Loading Protocols** — Supports ramp, hold, and multi-rate scenarios (S0–S3)
- **Fast Inference** — ~ms per specimen after training

## Results at a Glance

| Parameter Type | Identifiability | Example R² (D3) |
|----------------|-----------------|-----------------|
| Protocol-Robust | High | 0.82 – 0.96 |
| Protocol-Sensitive | Requires holds | 0.72 – 0.89 |
| Structurally Weak | Physics-limited | < 0.43 |




## Documentation

See [`README/`](README/) for complete technical documentation and user guide.



## Advisors

- Prof. Hans-Andrea Loeliger (ISI, ETH Zurich)
- Dr. Ehsan Hosseini (Empa)
- Haotian Xu (Empa)

