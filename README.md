# Continual Learning and Catastrophic Forgetting in Neural Networks

## Overview

This project presents a comprehensive Continual Learning (CL) framework developed to study the problem of **Catastrophic Forgetting** in neural networks and evaluate methods that enable models to learn tasks sequentially while retaining previously acquired knowledge.

The framework demonstrates catastrophic forgetting using a baseline sequential learning approach and implements two popular continual learning methods:

* Experience Replay
* Elastic Weight Consolidation (EWC)

The project further evaluates these methods across multiple continual learning benchmarks using standard continual learning metrics and generates publication-quality visualizations for analysis.

---

## Objectives

The primary objectives of this project are:

* Demonstrate catastrophic forgetting in neural networks.
* Implement continual learning techniques to reduce forgetting.
* Evaluate retention and transfer capabilities of different methods.
* Compare methods across multiple benchmark datasets.
* Generate reproducible experimental results and visualizations.

---

## Continual Learning Methods Implemented

### 1. Baseline Sequential Learning

The model is trained sequentially on tasks without any forgetting mitigation strategy.

Workflow:

Task 1 → Task 2 → Task 3 → ... → Task N

This serves as the reference model for demonstrating catastrophic forgetting.

---

### 2. Experience Replay

Experience Replay stores representative samples from previously learned tasks and reuses them during future training.

Training Dataset:

Current Task Data + Replay Memory

Benefits:

* Reduces forgetting
* Improves task retention
* Simple and effective continual learning strategy

---

### 3. Elastic Weight Consolidation (EWC)

EWC identifies important model parameters using the Fisher Information Matrix and penalizes changes to those parameters during future task learning.

EWC Loss:

L_total = L_current + λ Σ Fi(θi − θ*i)^2

Benefits:

* Memory-efficient
* Preserves important parameters
* Reduces forgetting through regularization

---

## Benchmarks Implemented

### Basic Benchmark

Task 1 → MNIST

Task 2 → Fashion-MNIST

Used for initial catastrophic forgetting demonstration.

---

### Permuted MNIST

Each task applies a unique random pixel permutation to the original MNIST images.

Example:

Task 1 → Permutation A

Task 2 → Permutation B

Task 3 → Permutation C

...

---

### Rotated MNIST

Each task applies a different image rotation angle.

Example:

Task 1 → 0°

Task 2 → 15°

Task 3 → 30°

Task 4 → 45°

...

---

## Model Architecture

A lightweight Convolutional Neural Network (CNN) is used throughout the experiments.

Architecture:

* Conv Layer 1
* ReLU
* Max Pooling
* Conv Layer 2
* ReLU
* Max Pooling
* Fully Connected Layer
* Output Layer

The architecture dynamically supports:

* MNIST
* Fashion-MNIST
* Permuted MNIST
* Rotated MNIST

through configurable input channels and output classes.

---

## Evaluation Metrics

### Average Accuracy (ACC)

Measures overall final task performance.

ACC = (1/T) Σ A(T,i)

Higher ACC indicates better overall performance.

---

### Backward Transfer (BWT)

Measures forgetting.

BWT = (1/(T−1)) Σ [A(T,i) − A(i,i)]

Interpretation:

* Positive → Improvement
* Zero → No Forgetting
* Negative → Forgetting

---

### Forward Transfer (FWT)

Measures knowledge transfer to future tasks.

FWT = (1/(T−1)) Σ [A(i−1,i) − b(i)]

Higher values indicate better transfer learning.

---

## Accuracy Matrix

The project generates a task-wise accuracy matrix:

A(i,j)

where:

* i = training task
* j = evaluation task

Example:

| Train/Eval | T1 | T2 | T3 |
| ---------- | -- | -- | -- |
| T1         | 98 | -  | -  |
| T2         | 76 | 97 | -  |
| T3         | 63 | 92 | 96 |

The matrix provides a complete view of forgetting and retention behavior.

---

## Project Structure

```text
continual_learning_project/

├── main.py

├── models/
│   └── cnn_model.py

├── benchmarks/
│   ├── basic_benchmark.py
│   ├── permuted_mnist.py
│   ├── rotated_mnist.py
│   └── task_manager.py

├── methods/
│   ├── sequential_trainer.py
│   ├── replay.py
│   ├── replay_buffer.py
│   ├── ewc.py
│   └── ewc_utils.py

├── experiments/
│   ├── experiment_runner.py
│   ├── benchmark_factory.py
│   ├── method_factory.py

├── metrics/
│   ├── accuracy_matrix.py
│   ├── accuracy.py
│   └── metrics_summary.py
│   └── backward_transfer.py
│   └── forward_transfer.py

├── utils/
│   ├── datasets.py
│   ├── plotting.py
│   └── result_saver.py

├── visualization/
│   ├── metric_plots.py
│   ├── retention_plots.py
│   ├── heatmaps.py
│   ├── summary_tables.py
│   └── visualization_runner.py

├── results/

├── visualizations/

└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd continual_learning_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```bash
torch
torchvision
numpy
pandas
matplotlib
seaborn
```

Install using:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn
```

---

## Running Experiments

Execute:

```bash
python main.py
```

The pipeline automatically:

1. Runs all benchmarks
2. Runs all methods
3. Computes ACC, BWT, FWT
4. Generates accuracy matrices
5. Saves results
6. Creates visualizations

---

## Generated Results

Experimental results are stored under:

```text
results/
```

Each experiment produces:

* accuracy_matrix.csv
* metrics.csv

---

## Generated Visualizations

Visualizations are stored under:

```text
visualizations/
```

Includes:

### Metric Comparison Charts

* ACC
* BWT
* FWT

Comparison among:

* Baseline
* Replay
* EWC

---

### Retention Curves

Task-wise retention analysis showing forgetting progression.

---

### Accuracy Heatmaps

Visual representation of accuracy matrices.

Color Mapping:

* Green → Strong Retention
* Yellow → Moderate Retention
* Red → High Forgetting

---

### Summary Tables

Publication-ready comparison tables:

* ACC Comparison
* BWT Comparison
* FWT Comparison

---

## Experimental Workflow

Benchmark Selection

↓

Task Generation

↓

Sequential Training

↓

Replay / EWC

↓

Accuracy Matrix Construction

↓

ACC / BWT / FWT Computation

↓

Result Storage

↓

Visualization Generation

↓

Research Analysis

---

## Key Outcomes

* Successfully reproduced catastrophic forgetting.
* Implemented Replay and EWC continual learning methods.
* Evaluated multiple continual learning benchmarks.
* Measured performance using standard CL metrics.
* Built an automated experimentation pipeline.
* Generated publication-quality visualizations.
* Developed a reusable continual learning research framework.

---

## Future Work

Potential future extensions include:

* Gradient Episodic Memory (GEM)
* Averaged GEM (A-GEM)
* Learning without Forgetting (LwF)
* Progressive Neural Networks
* Transformer-based Continual Learning
* Online Continual Learning
* Task-Free Continual Learning

---

## Author

Bharath Kumar Pidugu

Master's Student

Frankfurt University of Applied Sciences

Research Area: Continual Learning, Deep Learning, Artificial Intelligence
