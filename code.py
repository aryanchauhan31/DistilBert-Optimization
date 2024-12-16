pip install transformers datasets torch

from datasets import load_dataset
from transformers import AutoTokenizer

# Load MRPC and SST-2 datasets
mrpc_dataset = load_dataset("glue", "mrpc")
sst2_dataset = load_dataset("glue", "sst2")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['sentence1'] if 'sentence1' in examples else examples['sentence'],
                     examples['sentence2'] if 'sentence2' in examples else None,
                     truncation=True, padding="max_length", max_length=128)

mrpc_encoded = mrpc_dataset.map(tokenize_function, batched=True)
sst2_encoded = sst2_dataset.map(tokenize_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load the model for classification
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Prepare the dataset
mrpc_encoded = mrpc_encoded.rename_column("label", "labels")
mrpc_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./mrpc_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mrpc_encoded["train"],
    eval_dataset=mrpc_encoded["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Load the model for classification
model_sst2 = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Prepare the dataset
sst2_encoded = sst2_encoded.rename_column("label", "labels")
sst2_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define training arguments
training_args_sst2 = TrainingArguments(
    output_dir="./sst2_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./sst2_logs",
    logging_steps=10,
    load_best_model_at_end=True,
)

# Create Trainer
trainer_sst2 = Trainer(
    model=model_sst2,
    args=training_args_sst2,
    train_dataset=sst2_encoded["train"],
    eval_dataset=sst2_encoded["validation"],
    tokenizer=tokenizer,
)

# Train the model
trainer_sst2.train()

import torch

# Save MRPC model
torch.save(model.state_dict(), "./mrpc_model.pt")

# Save SST-2 model
torch.save(model_sst2.state_dict(), "./sst2_model.pt")
print("Models saved as .pt files.")

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)

with torch.no_grad():
    for example in mrpc_validation:
        # Move inputs and labels to the same device as the model
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}
        labels = torch.tensor([example["labels"]]).to(device)
        
        # Forward pass through the base model
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Add predictions and labels to the metric
        mrpc_metric.add_batch(predictions=predictions, references=labels)

# Compute the final metric
mrpc_results = mrpc_metric.compute()
print("Base Model MRPC Results:", mrpc_results)

from datasets import load_dataset
from transformers import AutoTokenizer

# Load the SST-2 dataset
sst2_dataset = load_dataset("glue", "sst2")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the SST-2 dataset
def tokenize_function(examples):
    return tokenizer(examples['sentence'], truncation=True, padding="max_length", max_length=128)

sst2_encoded = sst2_dataset.map(tokenize_function, batched=True)

# Prepare the validation dataset
sst2_encoded = sst2_encoded.rename_column("label", "labels")
sst2_encoded.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
sst2_validation = sst2_encoded["validation"]


from evaluate import load
import torch

# Load the metric for SST-2
sst2_metric = load("glue", "sst2")

# Ensure the base model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_sst2.to(device)

# Evaluate the base model
model_sst2.eval()

with torch.no_grad():
    for example in sst2_validation:
        # Move inputs and labels to the same device as the model
        inputs = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in example.items() if k in ["input_ids", "attention_mask"]}
        labels = torch.tensor([example["labels"]]).to(device)
        
        # Forward pass through the base model
        outputs = model_sst2(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Add predictions and labels to the metric
        sst2_metric.add_batch(predictions=predictions, references=labels)

# Compute the final metric
sst2_results = sst2_metric.compute()
print("Base Model SST-2 Results:", sst2_results)

import torch

# Load the MRPC and SST-2 models
model_mrpc = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model_mrpc.load_state_dict(torch.load("./mrpc_model.pt"))
model_mrpc.eval()

model_sst2 = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model_sst2.load_state_dict(torch.load("./sst2_model.pt"))
model_sst2.eval()

# Apply dynamic quantization to both models
quantized_model_mrpc = torch.quantization.quantize_dynamic(
    model_mrpc,  # Model to quantize
    {torch.nn.Linear},  # Specify layers to quantize
    dtype=torch.qint8  # Quantize weights to int8
)

quantized_model_sst2 = torch.quantization.quantize_dynamic(
    model_sst2,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save the quantized models
torch.save(quantized_model_mrpc, "./quantized_mrpc_model.pt")
torch.save(quantized_model_sst2, "./quantized_sst2_model.pt")

print("Quantized models saved successfully!")

import os

def get_file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)  # Convert bytes to MB

# Compare file sizes
original_mrpc_size = get_file_size("./mrpc_model.pt")
quantized_mrpc_size = get_file_size("./quantized_mrpc_model.pt")

original_sst2_size = get_file_size("./sst2_model.pt")
quantized_sst2_size = get_file_size("./quantized_sst2_model.pt")

print(f"Original MRPC model size: {original_mrpc_size:.2f} MB")
print(f"Quantized MRPC model size: {quantized_mrpc_size:.2f} MB")
print(f"Original SST-2 model size: {original_sst2_size:.2f} MB")
print(f"Quantized SST-2 model size: {quantized_sst2_size:.2f} MB")


# Tokenizer setup
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Define inference function
def inference(model, sentence, tokenizer):
    model.eval()
    device = torch.device("cpu")  # Quantized models typically run on CPU
    model.to(device)

    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    return predictions.item()

# Example sentences for inference
sentence_mrpc = "The quick brown fox jumps over the lazy dog."
sentence_sst2 = "The movie was absolutely fantastic!"

# Inference on MRPC
print("MRPC Inference:")
prediction_mrpc = inference(quantized_model_mrpc, sentence_mrpc, tokenizer)
print(f"Prediction for MRPC: {prediction_mrpc}")

# Inference on SST-2
print("SST-2 Inference:")
prediction_sst2 = inference(quantized_model_sst2, sentence_sst2, tokenizer)
print(f"Prediction for SST-2: {prediction_sst2}")


from torch.utils.data import DataLoader
import torch
from evaluate import load

# Prepare DataLoader for MRPC and SST-2
validation_loader_mrpc = DataLoader(mrpc_encoded["validation"], batch_size=16)
validation_loader_sst2 = DataLoader(sst2_encoded["validation"], batch_size=16)

# Initialize GLUE metrics
metric_mrpc = load("glue", "mrpc")
metric_sst2 = load("glue", "sst2")

# Ensure the quantized models are on the correct device
device = torch.device("cpu")  # Quantized models are typically run on CPU
quantized_model_mrpc.to(device)
quantized_model_sst2.to(device)

# Define evaluation function
def evaluate_model(model, dataloader, metric):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the correct device
            inputs = {k: batch[k].to(device) for k in ["input_ids", "attention_mask"]}
            labels = batch["labels"].to(device)

            # Perform inference
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

            # Add predictions and references to the metric
            metric.add_batch(predictions=predictions.cpu(), references=labels.cpu())

    return metric.compute()

# Evaluate MRPC
print("Evaluating MRPC...")
mrpc_results = evaluate_model(quantized_model_mrpc, validation_loader_mrpc, metric_mrpc)
print("Quantized MRPC Results:", mrpc_results)

# Evaluate SST-2
print("Evaluating SST-2...")
sst2_results = evaluate_model(quantized_model_sst2, validation_loader_sst2, metric_sst2)
print("Quantized SST-2 Results:", sst2_results)
