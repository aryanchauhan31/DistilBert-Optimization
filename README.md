
# **README.md**

## **Notebook Overview**

This Jupyter notebook automates the process of **fine-tuning, quantizing, and evaluating a DistilBERT model** on GLUE tasks (e.g., MRPC and SST-2). It uses the Hugging Face `transformers` library for model loading, training, and evaluation.

---

## **Dependencies**

Ensure you have the following libraries installed:

```bash
pip install transformers datasets torch evaluate
```

---

## **Table of Contents**

1. **Environment Setup**
2. **Data Loading and Preprocessing**
3. **Model Fine-Tuning**
4. **Model Quantization**
5. **Model Evaluation**
6. **Inference on Test Sentences**

---

## **Steps Explained**

### **1. Environment Setup**
- Install required libraries:
    ```python
    pip install transformers datasets evaluate
    ```

### **2. Data Loading and Preprocessing**
- The notebook loads the MRPC and SST-2 datasets using the Hugging Face `datasets` library:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("glue", "mrpc")
    ```

### **3. Model Fine-Tuning**
- Fine-tune a **DistilBERT** model for classification tasks:
    ```python
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    ```

- Training is performed using the `Trainer` API.

### **4. Model Quantization**
- Apply **dynamic quantization** to reduce model size and improve inference speed:
    ```python
    import torch
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    ```

- Save the quantized model:
    ```python
    torch.save(quantized_model, "./quantized_model.pt")
    ```

### **5. Model Evaluation**
- Evaluate the model on MRPC and SST-2 tasks using Hugging Face's `evaluate` library:
    ```python
    from evaluate import load
    metric = load("glue", "mrpc")
    ```

### **6. Inference**
- Perform inference on test sentences:
    ```python
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer("Sample text", return_tensors="pt")
    outputs = quantized_model(**inputs)
    predictions = torch.argmax(outputs.logits)
    ```

---

## **Outputs**

The following results are expected:
- **Model Metrics**:
    - Accuracy and F1 score for MRPC
    - Accuracy for SST-2
- **Example Predictions**:
    ```
    MRPC Prediction: 0
    SST-2 Prediction: 1
    ```

---

## **Author**
- Notebook created using Hugging Face `transformers` and `datasets`.
- Quantization performed using PyTorch.

---

## **Model Compression and Accuracy**

### **1. Baseline Model**
- **Model Size**: ~268 MB
- **Accuracy**:
    - MRPC: 87% (Accuracy), 89% (F1 Score)
    - SST-2: 92% (Accuracy)

### **2. Quantized Model**
- **Compressed Model Size**: ~96 MB
- **Accuracy**:
    - MRPC: 86% (Accuracy), 88% (F1 Score)
    - SST-2: 91% (Accuracy)

### **Compression Ratio**
- **MRPC**: ~2.79x smaller
- **SST-2**: ~2.79x smaller

The quantized models retain most of their accuracy while achieving significant compression.
