from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import mean_squared_error

# Load dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)

# Tokenize dataset
def preprocess_data(examples):
    # Tokenize the essays with truncation and padding
    return tokenizer(examples["essay"], truncation=True, padding="max_length", max_length=512)

# Apply preprocessing to the dataset
tokenized_data = dataset.map(preprocess_data, batched=True)

# Prepare datasets for PyTorch
train_data = tokenized_data["train"]
test_data = tokenized_data["test"]

# Convert labels (scores) to PyTorch tensors
train_data = train_data.map(lambda x: {"labels": torch.tensor(x["score"], dtype=torch.float32)})
test_data = test_data.map(lambda x: {"labels": torch.tensor(x["score"], dtype=torch.float32)})

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",                   # Directory to save model checkpoints
    evaluation_strategy="epoch",             # Evaluate at the end of each epoch
    learning_rate=2e-5,                       # Learning rate
    per_device_train_batch_size=8,            # Batch size per device
    num_train_epochs=3,                       # Number of epochs
    weight_decay=0.01,                        # Weight decay for regularization
    logging_dir="./logs",                     # Directory to save logs
    logging_steps=10                          # Log every 10 steps
)

# Define the trainer
trainer = Trainer(
    model=model,                              # Model for training
    args=training_args,                       # Training arguments
    train_dataset=train_data,                 # Training dataset
    eval_dataset=test_data,                   # Evaluation dataset
    tokenizer=tokenizer                       # Tokenizer for preprocessing
)

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating the model...")
predictions = trainer.predict(test_data)
predicted_scores = predictions.predictions.squeeze()
true_scores = np.array([example["score"] for example in test_data])

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(true_scores, predicted_scores)
print(f"Mean Squared Error: {mse}")

# Save the trained model and tokenizer
model.save_pretrained("./essay_scoring_model")
tokenizer.save_pretrained("./essay_scoring_model")
print("Model and tokenizer saved to './essay_scoring_model'")
