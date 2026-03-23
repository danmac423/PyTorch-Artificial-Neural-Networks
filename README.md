# Artificial Neural Networks (SSNE) - University Projects

This repository contains a collection of projects completed during the **Artificial Neural Networks (SSNE)** course. The projects cover a wide spectrum of machine learning tasks, including regression, classification, computer vision (GANs, image recognition), sequence processing (RNNs), and Natural Language Processing (NLP).

## Table of Contents

### 🏠 Project 2: Real Estate Price Prediction
**Objective:** Build a model to classify properties into one of three price categories: `cheap`, `average`, or `expensive` based on technical parameters and location.
* **Data:** Tabular dataset including features such as year built, square footage, floor level, and proximity to subway/bus stations.
* **Technologies:** PyTorch, PyTorch Lightning, Scikit-learn (for data scaling and class weights).
* **Model:** A Multi-Layer Perceptron (MLP) featuring `BatchNorm1d` and `Dropout` layers to prevent overfitting.
* **Challenge:** Handling highly imbalanced classes using balanced class weights in the loss function.

### 🖼️ Project 3: Image Classification (50 Classes)
**Objective:** Develop an image classifier capable of distinguishing between 50 different classes of objects and animals.
* **Data:** Organized image set loaded via `torchvision.ImageFolder`.
* **Technologies:** PyTorch, Torchvision, PyTorch Lightning.
* **Model:** A custom architecture utilizing **MBConv** (Mobile Inverted Bottleneck Convolution) blocks and **Squeeze-and-Excitation (SEBlock)** for efficient feature extraction.

### 🚦 Project 4: Traffic Sign Generation (Conditional DCGAN)
**Objective:** Implementation of a generative model to create synthetic traffic sign images conditioned on a specific class.
* **Data:** Traffic sign dataset consisting of 43 distinct classes.
* **Model:** A Conditional Deep Convolutional GAN (DCGAN). It uses embeddings to condition both the Generator (`ConvTranspose2d`) and the Discriminator (`Conv2d`).
* **Monitoring:** Integrated with **Weights & Biases** to track the visual quality of generated images across training epochs.

### 🎹 Project 5: Composer Classification (RNN/LSTM)
**Objective:** Recognize the composer (e.g., Bach, Beethoven, Debussy) based on a sequence of musical chords.
* **Data:** Variable-length musical sequences stored in `.pkl` format.
* **Model:** A **Bidirectional LSTM** network with an `Embedding` layer. The architecture uses `pad_sequence` to handle the varying lengths of input data.

### 🤬 Project 6: Hate Speech Detection (NLP)
**Objective:** Classify Polish text samples to detect the presence of hate speech.
* **Data:** Social media text data with binary labels.
* **Model:** Fine-tuned **trelbert** (a Polish-specific BERT model) using the `transformers` library and `WeightedLossTrainer` to account for class imbalance.

---

## File Structure
Each project directory typically includes:
* `*.ipynb`: Jupyter Notebooks containing data analysis, preprocessing, and training loops.
* `model.py`: Python scripts defining the neural network architectures.
* `pred.csv`: Output predictions for the final evaluation.
* `tresc.txt` / `tresc_zadania.txt`: Detailed project requirements provided by the instructor.

## Installation & Setup
This project uses the `uv` package manager for dependency management. To set up the environment:

```bash
uv sync
```

**Key Dependencies:**
* `torch` & `torchvision`
* `lightning` (PyTorch Lightning)
* `transformers` & `datasets` (Hugging Face)
* `pandas` & `numpy`
* `scikit-learn`
* `wandb` (Weights & Biases)
