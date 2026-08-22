# LIGO Glitch Classification with Deep Learning & Soft Voting

This repository contains machine learning pipelines for classifying gravitational wave glitch signals from the LIGO (Laser Interferometer Gravitational-Wave Observatory) dataset using PyTorch and Attention-based Convolutional Neural Networks (CNNs).

## Overview

Gravitational wave detectors like LIGO frequently pick up non-Gaussian noise transient signals called "glitches." This project automates the classification of these glitches across 22 distinct categories (e.g., *Whistle*, *Blip*, *Koi Fish*, *Violin Mode*) using spectrogram images (`url1` through `url4`) from the Gravity Spy dataset.

## Architecture

* **Base Model (`GravitySpyCNN1`)**: A hybrid model combining:
  * **3-Layer CNN Backbone**: Standard 2D Convolution, Batch Normalization, ReLU, and Max Pooling to extract spatial features from $64 \times 64$ grayscale spectrograms.
  * **Multihead Attention**: An 8-head self-attention module to capture temporal dependencies across sequence dimension features.
  * **Dense Classifier**: Fully connected layers with Dropout ($0.3$) mapping features to class logits.
* **Ensemble Method (Soft Voting)**: A multi-stream approach combining prediction probability vectors across multiple models trained on different spectrogram viewpoints (`url1CNN`, `url2CNN`, `url3CNN`, `url4CNN`).

## Repository Structure

```text
├── MLxLIGO_url2_mine.ipynb         # Training notebook for single URL stream CNN
├── Copy_of_MLxLIGO_SoftVoting.ipynb # Ensemble pipeline using Soft Voting across multiple trained models
└── README.md
