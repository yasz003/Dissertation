# Dissertation Repository: Physics-Informed Neural Networks for Optimising Energy Demand of Buildings

This repository contains the implementation and evaluation code for my BSc dissertation titled **"Physics-informed Neural Networks for Optimising Energy Demand of Buildings"**, submitted to the University of Manchester.

## Overview

Buildings contribute significantly to global energy consumption, and accurate thermal modeling is essential for improving energy efficiency. My dissertation explores the use of Physics-Informed Neural Networks (PiNNs), specifically using a PhysNet architecture, to predict indoor thermal dynamics. PiNNs integrate physics-based constraints from the Resistance-Capacitance (2R2C) thermal model into the learning process, offering both interpretability and accuracy.

The study evaluates PhysNet against a purely data-driven Neural Network (NN) by predicting indoor temperatures based on indoor/outdoor conditions and HVAC operations over four working days. Results indicate that PhysNet significantly outperforms the purely data-driven approach, achieving a lower Mean Absolute Error (MAE).

## Repository Structure

- **PiNN Implementation:**
  - `PhysNet.py`: Contains the full implementation of the PhysNet model, integrating the 2R2C thermal model.

- **Data-Driven NN Implementation:**
  - `DataDrivenNN.py`: Adaptation of PhysNet with the physical constraints removed for direct performance comparison.

- **Training & Evaluation:**
  - `train_model.py`: Script for training both models.
  - `compute_mae.py`: Computes and compares the Mean Absolute Error (MAE) of the predictions from both models.

- **Figures & Results:**
  - `figures.py`: Generates visual comparisons of predicted and actual indoor temperatures, illustrating performance differences between PhysNet and the data-driven NN.

## Key Results

- **Mean Absolute Error (MAE):**
  - PhysNet (PiNN): 0.54°C
  - Data-driven NN: 0.74°C

PhysNet's enhanced performance demonstrates the value of embedding physics-based constraints in neural networks for building thermal management, particularly beneficial when datasets are limited.

## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/yasz003/Dissertation.git
   ```

2. Train the models:
   ```bash
   python train_model.py
   ```

3. Evaluate predictions:
   ```bash
   python compute_mae.py
   ```

4. Generate comparison figures:
   ```bash
   python figures.py
   ```

## Future Work

This study highlighted opportunities for improvement, including using larger datasets, enhancing sensor coverage, and testing PhysNet in more complex scenarios like multi-zone environments.

## Contact

- **Author:** Yacine Benhamed
- **University:** University of Manchester
- **Email:** [Your email address]

## Acknowledgements

This research was funded by Future Energy Associates.

---

**Note:** Ensure to adjust file paths, dependencies, and scripts as necessary based on your system setup.

