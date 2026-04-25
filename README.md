# Intro-to-Machine-Learning-Homework

P/s: This repo is created to show my works to Prof. Ngo Van Linh. The purpose is to upload my solutions to every homework problem, encompassing Hw 1 to Hw 18. I really appreciate my dear professor to give me a chance at his training course.

This repository contains a comprehensive collection of machine learning and deep learning assignments completed over an 18-week training program. Each week focuses on a specific set of algorithms, progressing from foundational mathematics to advanced AI architectures.

## Table of Contents

- [Week 1: Linear Regression from Scratch](#week-1-linear-regression-from-scratch)
- [Week 2: Random Forest Implementation](#week-2-random-forest-implementation)
- [Week 3: Naive Bayes Classification](#week-3-naive-bayes-classification)
- [Week 4: Decision Tree Foundations](#week-4-decision-tree-foundations)
- [Week 5: Unsupervised Learning - K-Means](#week-5-unsupervised-learning---k-means)
- [Week 6: Advanced Linear Regression Analysis](#week-6-advanced-linear-regression-analysis)
- [Week 7: Random Forest Evaluation & Metrics](#week-7-random-forest-evaluation--metrics)
- [Week 8: MLP in NumPy](#week-8-mlp-in-numpy)
- [Week 9: Introduction to PyTorch MLP](#week-9-introduction-to-pytorch-mlp)
- [Week 10: Deep MLP & Model Persistence](#week-10-deep-mlp--model-persistence)
- [Week 11: CNNs (LeNet-5 & DenseNet)](#week-11-cnns-lenet-5--densenet)
- [Week 12: Advanced Data Augmentation](#week-12-advanced-data-augmentation)
- [Week 13: Model Ensembling (CIFAR-10)](#week-13-model-ensembling-cifar-10)
- [Week 14: Sequence Models (RNN & GRU)](#week-14-sequence-models-rnn--gru)
- [Week 15: Transformer Infrastructure](#week-15-transformer-infrastructure)
- [Week 16: Generative Adversarial Networks (DCGAN)](#week-16-generative-adversarial-networks-dcgan)
- [Week 17: Graph Neural Networks (DeepWalk)](#week-17-graph-neural-networks-deepwalk)
- [Week 18: Deep Q-Learning (DQN)](#week-18-deep-q-learning-dqn)

---

## Week 1: Linear Regression from Scratch
**File**: `HW1_LVD.ipynb`
- **Requirements**: Implement basic Linear Regression using Gradient Descent.
- **Implementation**: Manual derivation of loss function gradients and weight updates using NumPy.
- **Insights**: Understanding the role of learning rate in convergence and the impact of feature scale on training speed.

## Week 2: Random Forest Implementation
**File**: `HW2_LVD.ipynb`
- **Requirements**: Build a Random Forest classifier from scratch.
- **Implementation**: Bootstrap aggregating (Bagging) and random feature selection for building multiple trees.
- **Insights**: Random Forests effectively reduce variance and prevent overfitting compared to a single decision tree.

## Week 3: Naive Bayes Classification
**File**: `HW3_LVD.ipynb`
- **Requirements**: Implement the Gaussian Naive Bayes algorithm.
- **Implementation**: Calculated class priors and conditional probabilities using Gaussian distribution assumptions.
- **Insights**: Naive Bayes serves as an excellent baseline due to its simplicity and efficiency, especially for high-dimensional data.

## Week 4: Decision Tree Foundations
**File**: `HW4_LVD.ipynb`
- **Requirements**: Build a basic Decision Tree classifier.
- **Implementation**: Recursive splitting based on Information Gain and Entropy criteria.
- **Insights**: Deep trees are prone to overfitting; pruning or limiting depth is crucial for generalization.

## Week 5: Unsupervised Learning - K-Means
**File**: `HW5_LVD.ipynb`
- **Requirements**: Implement the K-Means clustering algorithm.
- **Implementation**: Centroid initialization, assignment step, and update step until convergence.
- **Insights**: K-Means is sensitive to the initial placement of centroids (solved by methods like K-Means++).

## Week 6: Advanced Linear Regression Analysis
**File**: `HW6_LVD.ipynb`
- **Requirements**: Compare custom Linear Regression with Scikit-Learn's implementation.
- **Implementation**: Analysis of residuals and model performance metrics (MSE, R-squared).
- **Insights**: Validating custom logic against optimized libraries confirms the correctness of the scratch implementation.

## Week 7: Random Forest Evaluation & Metrics
**File**: `HW7_LVD.ipynb`
- **Requirements**: Perform detailed performance analysis on Random Forest.
- **Implementation**: Evaluation using Confusion Matrices, Precision, Recall, and F1-score.
- **Insights**: Metrics like F1-score provide a more balanced view of performance than accuracy, especially on imbalanced datasets.

## Week 8: MLP in NumPy
**File**: `HW8_LVD.ipynb`
- **Requirements**: Create a Multi-Layer Perceptron using only NumPy.
- **Implementation**: Manual implementation of forward propagation, backpropagation, and activation functions like Sigmoid.
- **Insights**: Scratch implementation is the best way to understand the complex flow of gradients during training.

## Week 9: Introduction to PyTorch MLP
**File**: `HW9_LVD.ipynb`
- **Requirements**: Port the MNIST classification task to PyTorch.
- **Implementation**: Using `nn.Module` and `optim` for streamlined network definition and training.
- **Insights**: Transitioning to frameworks drastically reduces boilerplate code while increasing computational efficiency through automatic differentiation.

## Week 10: Deep MLP & Model Persistence
**File**: `HW10_LVD.ipynb`
- **Requirements**: Build a deeper MLP and implement model saving/loading.
- **Implementation**: 3+ hidden layers training on MNIST with checkpointing logic (`torch.save`).
- **Insights**: Increasing depth can improve accuracy but requires careful tuning of learning rates and regularization to avoid divergence.

## Week 11: CNNs (LeNet-5 & DenseNet)
**Files**: `HW11.1_LVD.ipynb`, `HW11.2_LVD.ipynb`
- **Requirements**: Implement classical and modern CNN architectures.
- **Implementation**: LeNet-5 for Fashion-MNIST and structural implementation of DenseBlocks for DenseNet.
- **Insights**: Feature concatenation in DenseNet improves gradient flow and allows for parameter efficiency compared to simple stacking.

## Week 12: Advanced Data Augmentation
**File**: `HW12_LVD.ipynb`
- **Requirements**: Implement robust augmentation pipelines for image data.
- **Implementation**: Used `imgaug` for spatial transformations (Rotate, Affine) and noise injection (Gaussian Blur).
- **Insights**: Augmentation is a powerful regularization technique that simulates larger datasets and improves model robustness.

## Week 13: Model Ensembling (CIFAR-10)
**File**: `HW13_LVD.ipynb`
- **Requirements**: Combine multiple models to improve accuracy on CIFAR-10.
- **Implementation**: Soft-voting mechanism to average prediction probabilities across multiple CNNs.
- **Insights**: Ensembling mitigates individual model errors and provides a consistent performance boost.

## Week 14: Sequence Models (RNN & GRU)
**Files**: `HW14.1_LVD.ipynb`, `HW14.2_LVD.ipynb`
- **Requirements**: Implement Bi-directional RNNs and GRUs.
- **Implementation**: Image-as-sequence classification with Bi-RNN and language modeling using Gated Recurrent Units from scratch.
- **Insights**: Gating mechanisms in GRUs are vital for handling long-term dependencies that simple RNNs struggle to learn.

## Week 15: Transformer Infrastructure
**File**: `HW15_LVD.ipynb`
- **Requirements**: Set up the training loop for a sequence-to-sequence Transformer.
- **Implementation**: Implementation of Teacher Forcing and causal masking logic in the training pipeline.
- **Insights**: Proper masking is essential for Transformers to ensure the decoder remains autoregressive.

## Week 16: Generative Adversarial Networks (DCGAN)
**File**: `HW16_LVD.ipynb`
- **Requirements**: Train a Deep Convolutional GAN to generate MNIST digits.
- **Implementation**: Min-max game between a Generator (upsampling) and a Discriminator (strided convolutions).
- **Insights**: GAN training stability relies on balanced learning rates and specific architecture choices (e.g., LeakyReLU, BatchNorm).

## Week 17: Graph Neural Networks (DeepWalk)
**File**: `HW17_LVD.ipynb`
- **Requirements**: Learn structural representations of graph nodes.
- **Implementation**: DeepWalk random walks combined with a custom Skip-gram (Word2vec) implementation.
- **Insights**: Mapping graph nodes to continuous vector spaces allows standard ML tools to be used for complex network analysis.

## Week 18: Deep Q-Learning (DQN)
**File**: `HW18_LVD.ipynb`
- **Requirements**: Solve the CartPole environment using Reinforcement Learning.
- **Implementation**: DQN agent with Experience Replay and a Target Network for stable Q-value estimation.
- **Insights**: Replay buffers and target networks are critical for stabilizing RL training by decorrelating experiences.
