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
- Encoded sequences to fixed length (`max_len = 256`). Also known as sequence length.

### ✅ Dataset & DataLoader
- Created a `TextClassificationDataset` class to return `(input_ids, label)` pairs.
- Split data into training and validation sets.
- Batched data using PyTorch `DataLoader`.


## 🧱 Model Architecture

This project includes two custom-built text classification models using PyTorch for a 3-class problem: Model A wins, Model B wins, or Tie. Both models operate on sequences of fixed length (256 tokens) with an embedding dimension of 128 and batch size of 16.

---

### 🔹 Baseline Model: `LLM_Rater`

A simple feedforward model with no regularization.

**Input Dimensions:**
- `batch_size = 16`
- `sequence_length = 256`
- `vocab_size ≈ 20,000`  
- `embed_dim = 128`
- `hidden_dim = 64`

**Architecture:**
1. **Embedding Layer**  
   `nn.Embedding(vocab_size, embed_dim, padding_idx=0)`  
   → Output shape: `(batch_size, sequence_length, embed_dim)` → `(16, 256, 128)`

2. **Mean Pooling**  
   Averages across tokens  
   → Output shape: `(batch_size, embed_dim)` → `(16, 128)`

3. **Fully Connected Layer 1**  
   `nn.Linear(embed_dim, hidden_dim)` → `(128 → 64)`  
   → Output shape: `(16, 64)`

4. **ReLU Activation**

5. **Fully Connected Layer 2 (Output Layer)**  
   `nn.Linear(hidden_dim, num_classes)` → `(64 → 3)`  
   → Output shape: `(16, 3)`

**Total Parameters**: ~257K  
**Regularization**: None  

---

### 🔹 Regularized Model: `LLM_Rater_2` (TextCNN-inspired)

Same structure as the baseline with added **Dropout** layers to reduce overfitting.

**Changes from baseline:**
- Dropout added after mean pooling and after first linear layer
- Dropout rate: `0.2`

**Architecture:**
1. **Embedding Layer**  
   `nn.Embedding(vocab_size, embed_dim, padding_idx=0)`  
   → Output shape: `(batch_size, sequence_length, embed_dim) → (16, 256, 128)`

2. **Mean Pooling**  
   Averages across tokens
   → Output shape: `(batch_size, embed_dim) → (16, 128)`

4. **Dropout Layer 1**  
   `nn.Dropout(p=dropout_rate)`
   → Output shape: `(16, 128)`

6. **Fully Connected Layer 1**  
   `nn.Linear(embed_dim, hidden_dim)` → `(128 → 64)`  
   → Output shape: `(16, 64)`

7. **ReLU Activation**

8. **Dropout Layer 2**  
   → Output shape: `(16, 64)`

9. **Fully Connected Layer 2 (Output Layer)**  
   `nn.Linear(hidden_dim, num_classes)` → `(64 → 3)`  
   → Output shape: `(16, 3)`

**Total Parameters**: ~257K  
**Regularization**: Dropout (rate = 0.2)
---
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

## 📊 Baseline Model Output

| Class A | Tie   | Class B |
|---------|-------|---------|
| 0.152   | 0.202 | 0.646   |
| 0.671   | 0.299 | 0.029   |
| 0.208   | 0.723 | 0.068   |

**Predicted Class Labels:** 
- `2` → Model Tie  
- `0` → Winner A  
- `1` → Winner B  

---

## 📊 TextCNN Model Ouptput

**Probability Scores (Softmax Output):**

| Winner A | Winner B | Tie   |
|----------|----------|-------|
| 0.082    | 0.376    | 0.542 |
| 0.841    | 0.156    | 0.003 |
| 0.188    | 0.809    | 0.003 |

**Predicted Class Labels:**
- `2` → Model Tie  
- `0` → Winner A  
- `1` → Winner B  

---

### 📈 What Do These Results Indicate?

| Metric                  | Base Model | TextCNN Model |
|-------------------------|------------|----------------|
| **Train Accuracy**      | 89%        | 75%            |
| **Validation Accuracy** | ~Same      | ~Same          |
| **Validation Loss**     | 4.7        | 2.5            |

- ✅ **Lower Train Accuracy** indicates **reduced overfitting**. The TextCNN is less likely to memorize training data and more likely to generalize to unseen data.
- ✅ **Lower Validation Loss** means the model's probability estimates are more **confident and calibrated**, even if raw accuracy hasn’t improved.
- ➡️ **Flat Validation Accuracy** suggests we reduced overfitting, but might need additional tuning or data augmentation to improve further.

Overall, **the TextCNN model shows more stable and generalizable learning**, which is a sign of improved model quality.

---

## 🧠 Recommendations

- Consider using **pretrained transformer models** such as:
  - `DistilBERT`
  - `TinyBERT`
  - `MobileBERT`
  - `BERT-base`

  These models offer **significantly better language understanding**, but come at the **cost of compute and memory**. For lightweight tasks or limited compute environments, models like **TextCNN** can be fine-tuned and optimized further.

- Additional improvement ideas:
  - 🧪 **Regularization:** Try adding `Weight Decay` to penalize large weights.
  - 🛑 **Early Stopping:** Halt training when validation loss increases
  - 🔍 **Hyperparameter Tuning:** Explore different batch sizes, learning rates, and hidden dimensions
  - 🔄 **Data Augmentation:** Improve generalization by increasing diversity of training examples
---  
Final Conclusion: Transformer models are better suited for this task because they are designed to capture contextual relationships across different parts of a sequence. Even when combining the user prompt and multiple model outputs into a single input, transformers can understand their roles if special tokens or structured prompts are used. Unlike simpler models, they use self-attention to compare and relate information across the full input, enabling deeper understanding of which output aligns better with the prompt. This makes them ideal for tasks where distinguishing subtle differences in relevance and coherence is crucial.


