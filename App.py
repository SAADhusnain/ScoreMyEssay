from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
from spellchecker import SpellChecker

# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('cmudict')

# Load the dataset from the CSV file
data = pd.read_csv("Processed_data.csv")

# Assign the loaded data to the DataFrame
df = data

# Ensure the necessary columns exist in the DataFrame
required_columns = ['char_count', 'word_count', 'sent_count', 'avg_word_len', 
                     'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 
                     'adv_count', 'final_score']

if not all(col in df.columns for col in required_columns):
    print("Error: The following required columns are missing from the DataFrame:")
    print([col for col in required_columns if col not in df.columns])
    exit()

# Feature Engineering
def calculate_readability(avg_word_len, word_count, sent_count):
    return 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (avg_word_len / word_count)

def calculate_content_diversity(noun_count, adj_count, verb_count, adv_count, total_word_count):
    unique_parts_of_speech = noun_count + adj_count + verb_count + adv_count
    return unique_parts_of_speech / total_word_count

# Calculate new features and handle potential division by zero errors
df['readability_score'] = df.apply(lambda row: calculate_readability(row['avg_word_len'], row['word_count'], row['sent_count']) 
                                    if row['word_count'] != 0 and row['sent_count'] != 0 else 0, axis=1)

df['content_diversity'] = df.apply(lambda row: calculate_content_diversity(row['noun_count'], row['adj_count'], 
                                                                            row['verb_count'], row['adv_count'], row['word_count']) 
                                    if row['word_count'] != 0 else 0, axis=1)

### Hyperparameter Tuning Section ###

# Define hyperparameter tuning space
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 5, 10]
}

# Initialize RandomForestRegressor
rf = RandomForestRegressor(random_state=42)

# Perform GridSearchCV
X = df[['char_count', 'word_count', 'sent_count', 'avg_word_len', 'spell_err_count', 
         'noun_count', 'adj_count', 'verb_count', 'adv_count', 'readability_score', 'content_diversity']]
y = df['final_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best-performing model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print("Best Parameters:", best_params)

### Train with Best Parameters and Evaluate ###

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("Model Evaluation (Best Parameters):")
print("Mean Absolute Error: ", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R-Squared: ", r2_score(y_test, y_pred))

### Simulate Multiple Epochs with Early Stopping ###

# Define early stopping criteria
patience = 50  # Number of epochs with no improvement to wait before stopping
min_delta = 0.001  # Minimum change in metric to qualify as improvement

# Initialize lists to track performance over epochs
train_mae_history = []
test_mae_history = []
epoch = 0
best_mae = float('inf')
best_model_epoch = None
no_improvement_count = 0

while epoch < 20:  # Arbitrary max epochs for demonstration
    best_model.fit(X_train, y_train)
    
    # Predict on both train and test sets
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    # Calculate Mean Absolute Error (MAE) for both sets
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Track performance
    train_mae_history.append(train_mae)
    test_mae_history.append(test_mae)
    
    # Check for improvement on test set
    if test_mae < best_mae - min_delta:
        best_mae = test_mae
        best_model_epoch = epoch
        no_improvement_count = 0
    else:
        no_improvement_count += 1
    
    # Print current epoch's performance
    print(f"Epoch {epoch+1}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    # Check early stopping criteria
    if no_improvement_count >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement in {patience} epochs.")
        break
        
    epoch += 1

# Print best epoch's details
print(f"Best Epoch: {best_model_epoch+1}, Best Test MAE: {best_mae:.4f}")

### Function to Extract Features from Essay Text ###

def extract_features_from_text(text):
    spell = SpellChecker()
    
    # Tokenize text into words and sentences
    words = word_tokenize(text)
    sents = sent_tokenize(text)
    
    # Calculate basic features
    char_count = len(text)
    word_count = len(words)
    sent_count = len(sents)
    
    # Calculate average word length
    total_word_len = sum(len(word) for word in words)
    avg_word_len = total_word_len / word_count if word_count > 0 else 0
    
    # Calculate spell error count
    misspelled = spell.unknown(words)
    spell_err_count = len(misspelled)
    
    # Calculate parts of speech counts (simplified, assuming nouns, adjectives, verbs, and adverbs)
    # NOTE: This is a simplified approach and may not be entirely accurate.
    # For more accurate results, consider using a dedicated POS tagging library like NLTK's averaged perceptron tagger.
    noun_count = sum(1 for word in words if word.lower() in ['the', 'a', 'an', 'and'] or word.endswith('ion') or word.endswith('ment'))
    adj_count = sum(1 for word in words if word.endswith('able') or word.endswith('al') or word.endswith('ful') or word.endswith('less'))
    verb_count = sum(1 for word in words if word.endswith('ate') or word.endswith('ify') or word.endswith('ize'))
    adv_count = sum(1 for word in words if word.endswith('ly'))
    
    # Calculate readability score and content diversity
    readability_score = calculate_readability(avg_word_len, word_count, sent_count) if word_count != 0 and sent_count != 0 else 0
    content_diversity = calculate_content_diversity(noun_count, adj_count, verb_count, adv_count, word_count) if word_count != 0 else 0
    
    return {
    'char_count': char_count,
    'word_count': word_count,
    'sent_count': sent_count,
    'avg_word_len': avg_word_len,
    'spell_err_count': spell_err_count,
    'noun_count': noun_count,
    'adj_count': adj_count,
    'verb_count': verb_count,
    'adv_count': adv_count,
    'readability_score': readability_score,
    'content_diversity': content_diversity
}


### Command-Line Essay Scoring ###
### Command-Line Essay Scoring ###

def score_essay():
    print("Enter the essay text:")
    text = input()
    
    features = extract_features_from_text(text)
    essay_data = pd.DataFrame([features])

    
    prediction = best_model.predict(essay_data)
    
    print("Predicted Score: ", prediction[0])
    print("Breakdown:")
    print("  * Readability Score: ", features['readability_score'])
    print("  * Content Diversity: ", features['content_diversity'])
    print("  * Spell Error Count: ", features['spell_err_count'])
    print("  * Average Word Length: ", features['avg_word_len'])
    print("  * Word Count: ", features['word_count'])
    print("  * Sentence Count: ", features['sent_count'])
    print("  * Character Count: ", features['char_count'])
    print("  * Noun Count: ", features['noun_count'])
    print("  * Adjective Count: ", features['adj_count'])
    print("  * Verb Count: ", features['verb_count'])
    print("  * Adverb Count: ", features['adv_count'])

while True:
    print("\nOptions:")
    print("1. Score an Essay")
    print("2. Exit")
    choice = input("Enter your choice: ")
    
    if choice == "1":
        score_essay()
    elif choice == "2":
        break
    else:
        print("Invalid choice. Please choose again.")
1

