# 🔍 LLM Rating System – Text Classification

This repository implements a text classification system to evaluate and rate LLM responses using a lightweight, custome PyTorch model. The goal was to compare response quality between different language models and predict which response is better — Model A, Model B, or a Tie — without relying on large transformer architectures.

---

## 📌 Project Overview

- **Task:** Multi-class classification (Model A wins / Model B wins / Tie)  
- **Input:** Prompt + Response A + Response B (combined as a single input)  
- **Output:** Predicted class (`0` = A wins, `1` = B wins, `2` = Tie)  
- **Models Used:**  
  - Custom Feedforward Neural Network  
- **Frameworks:** PyTorch, NumPy, Pandas, Matplotlib 
- **Dataset Size:** ~175MB  

---

## 🧱 Project Structure

### ✅ Data Preprocessing
- Cleaned and decoded text fields (`prompt`, `response_a`, `response_b`).
- Tokenized using simple whitespace-based tokenization.
- Constructed a vocabulary of the 20,000 most common words + `<PAD>` and `<UNK>`.
- Encoded sequences to fixed length (`max_len = 256`).

### ✅ Dataset & DataLoader
- Created a `TextClassificationDataset` class to return `(input_ids, label)` pairs.
- Split data into training and validation sets.
- Batched data using PyTorch `DataLoader`.

### ✅ Model Architectures

1. **Baseline Feedforward Model**
   - Embedding → Mean Pooling → Linear → ReLU → Linear → Output
   - ~260k trainable parameters
   - 
### ✅ Training Loop
- Used Adam optimizer and CrossEntropy loss.
- Included validation after every epoch.
- Stored loss and accuracy metrics for both training and validation.

### ✅ Inference
- Returned softmax probabilities per class.
- Used `argmax` for final prediction.
- Tabulated predictions for easier interpretation.

---

## 📊 Example Output

| Class A | Tie   | Class B |
|---------|-------|---------|
| 0.152   | 0.202 | 0.646   |
| 0.671   | 0.299 | 0.029   |
| 0.208   | 0.723 | 0.068   |

Predicted Classes: `[2, 0, 1]` → `[Class B, Class A, Tie]`

---


