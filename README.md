# Imbalanced Data: Random Oversampling and SMOTE

## Applied Machine Learning Project

This project demonstrates how oversampling techniques can be used to handle
imbalanced datasets in machine learning.

The project focuses on two techniques:

- Random Oversampling (ROS)
- SMOTE (Synthetic Minority Over-sampling Technique)

## Project Overview

In an imbalanced dataset, one class contains significantly more samples than
another. This can cause a machine learning model to favor the majority class
and perform poorly on minority-class examples.

Random Oversampling balances the dataset by duplicating existing minority
samples.

SMOTE creates new synthetic minority samples by interpolating between
neighboring minority data points.

The project includes an animated visualization showing how these techniques
modify the dataset and affect minority-class representation.

## Animation

The animation demonstrates:

1. The imbalanced data problem
2. Why accuracy can be misleading
3. Oversampling
4. Random Oversampling
5. SMOTE
6. Random Oversampling vs SMOTE
7. Limitations of SMOTE
8. Correct train/test usage
9. Real-world applications
10. Final comparison and summary

## How SMOTE Works

A synthetic sample is generated using:

x_new = x_i + λ(x_nn - x_i)

where:

- x_i = minority sample
- x_nn = one of its nearest minority neighbors
- λ = random value between 0 and 1

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
