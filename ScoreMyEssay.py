from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

# Load dataset (ASAP AES or your essay scoring dataset)
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Tokenize dataset
def preprocess_data(examples):
    return tokenizer(examples["essay"], truncation=True, padding="max_length", max_length=512)

tokenized_data = dataset.map(preprocess_data, batched=True)

# Prepare datasets for PyTorch
train_data = tokenized_data["train"]
test_data = tokenized_data["test"]

# Convert labels to PyTorch tensors
train_data = train_data.map(lambda x: {"labels": torch.tensor(x["score"], dtype=torch.float32)})
test_data = test_data.map(lambda x: {"labels": torch.tensor(x["score"], dtype=torch.float32)})

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
predictions = trainer.predict(test_data)
predicted_scores = predictions.predictions.squeeze()
true_scores = np.array([example["score"] for example in test_data])

mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error: {mse}")

# Save the model
model.save_pretrained("./essay_scoring_model")
tokenizer.save_pretrained("./essay_scoring_model")
