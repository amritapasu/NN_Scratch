# Building a Neural Network from Scratch

## Overview
A configurable 3-layer feedforward neural network implemented in Java. Supports training via backpropagation or running with fixed weights. Uses sigmoid activation and is designed for learning Boolean functions from small datasets.

## Features

- 4-layer architecture: Input → Hidden1 → Hidden2 → Output  
- Sigmoid activation with backpropagation  
- Configurable via control file  
- Random or stored weight initialization  
- Saves training weights periodically  

## Results

- Successfully trained the network on Boolean operator datasets and **finger-count classification with 95% accuracy**.  
- The network consistently converges within specified iteration limits, minimizing error below defined thresholds.  
- Training time is efficiently tracked, enabling performance evaluation on different configurations.
