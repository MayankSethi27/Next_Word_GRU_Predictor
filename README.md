#  Next-Word Prediction using GRU (Gated Recurrent Unit)

This project implements a deep learning-based next-word predictor using a **GRU (Gated Recurrent Unit)** model. It is trained on a textual dataset to learn the sequence of words and predict the most probable next word based on the input context. This project demonstrates the application of **Recurrent Neural Networks** in **Natural Language Processing (NLP)** for sequence modeling tasks.

---

##  Table of Contents

- [ Project Overview](#-project-overview)
- [ Model Architecture](#-model-architecture)
- [ Installation & Setup](#️-installation--setup)
- [ How to Run](#-how-to-run)
- [ Evaluation Metrics](#-evaluation-metrics)
- [🛠 Technologies Used](#️-technologies-used)


---

##  Project Overview

Next-word prediction is a classic NLP task where the goal is to predict the next word in a sentence based on the previous words. This project:
- Cleans and tokenizes the input text.
- Converts sequences into a format suitable for training.
- Trains a GRU-based model to understand word sequences.
- Outputs predictions using trained weights.

---

##  Model Architecture

- **Embedding Layer:** Converts integer word indices to dense vectors of fixed size.
- **GRU Layer(s):** Captures temporal dependencies in sequences using gated recurrence.
- **Dense Output Layer:** Outputs a probability distribution over the vocabulary using softmax.

```python
Model Summary:
Embedding (input_dim=vocab_size, output_dim=embedding_dim)
→ GRU(units=hidden_units, return_sequences=False)
→ Dense(units=vocab_size, activation='softmax')
```
## Preprocessing Steps:
Lowercasing text

Removing punctuation and special characters

Tokenization using Tokenizer from Keras

Padding sequences to uniform length
---
##Installation & Setup
### 1.Clone the repository:

```bash
git clone https://github.com/MayankSethi27/next-word_gru_predictor.git
cd next-word_gru_predictor
```
### 2.Install dependencies:
```bash
pip install -r requirements.txt
```
---
##Evaluation Metrics
Loss Function: Categorical Crossentropy

Optimizer: Adam

Training Metrics:

Training Accuracy

Validation Accuracy

---
## Technologies Used
Python

TensorFlow / Keras

Numpy & Pandas

NLTK
