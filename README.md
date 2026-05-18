# EIS Fitting Framework

**Equivalent Circuit Model fitting for EIS data — Powered by LASII**

This repository contains a modular framework for **Electrical Impedance Spectroscopy (EIS)** data processing, visualization, validation, and equivalent circuit model (ECM) fitting.

The framework provides utilities for:

- Loading and organizing EIS datasets
- Nyquist and frequency-domain visualization
- Linear Kramers–Kronig (linKK) validation
- Equivalent Circuit Model fitting
- Parameter optimization and evaluation

---

# 0. Table of Contents

1. [Repository Structure](#1-repository-structure)
2. [Methodology](#2-methodology)
3. [Dataset](#3-dataset)
4. [Getting Started](#4-getting-started)
5. [Dependencies](#5-dependencies)
6. [Contributors](#6-contributors)

---

# 1. Repository Structure

```text
EIS_fit_PPGQ/
│
├── Examples/
│   ├── CircuitEquivalent.py
│   ├── LinearKK.py
│   ├── Plot_data.py
│   ├── fit_nyquist.py
│   └── load_data.py
│
├── utils/
│   ├── ECM_utils.py
│   ├── data_types.py
│   ├── file_utils.py
│   ├── fitting_utils.py
│   └── linKK.py
│
├── data/
│
├── fit_impedance_data.py
├── requirements.txt
└── .gitignore
```

## Folder Description

| Folder/File | Description |
|---|---|
| `Examples/` | Example scripts demonstrating EIS loading, plotting, linKK validation, and ECM fitting |
| `utils/` | Core utility functions and fitting framework modules |
| `data/` | Impedance data |
| `fit_impedance_data.py` | Main script for impedance fitting workflow |
| `requirements.txt` | Python dependencies required for the project |

---

# 2. Methodology

The framework follows a standard EIS processing workflow:

1. Import impedance spectroscopy data
2. Organize the dataset into structured objects
3. Perform preprocessing and consistency verification by Linear Kramers–Kronig (linKK) validation
5. Define the Equivalent Circuit Model (ECM)
6. Fit ECM parameters using optimization methods
7. Visualize results in Nyquist and frequency-domain representations

## Main Features

- Modular ECM fitting structure
- Nyquist plot visualization
- Frequency-domain analysis
- Linear Kramers–Kronig validation tools
- Optimization utilities for nonlinear fitting
- Flexible data handling utilities

---

# 3. Dataset

The repository includes a `data/` folder intended for:

- Experimental EIS datasets
- Validation datasets
- Synthetic impedance spectra for testing
- Suport extensions '.csv' and '.xlsx'

Typical supported measurements include:

- Frequency (`Hz`)
- Real impedance (`Z'`)
- Imaginary impedance (`Z''`)

---

# 4. Getting Started

## Clone the repository

```bash
git clone https://github.com/your_username/EIS_fit_PPGQ.git
cd EIS_fit_PPGQ
```

## Install dependencies

```bash
pip install -r requirements.txt
```
## upload Data

## Run example scripts

```bash
python Examples/fit_nyquist.py
```

or

```bash
python Examples/LinearKK.py
```

---

# 5. Dependencies

The following libraries were applied in this work:

```python
matplotlib   # data visualization
numpy        # numerical processing
scipy        # optimization and signal processing
impedance    # EIS analysis tools
sympy        # symbolic mathematics
```

Install manually if necessary:

```bash
pip install matplotlib numpy scipy impedance sympy
```

---

# 6. Contributors

- @EvertonTrentoJR
- @YamadaMuller
