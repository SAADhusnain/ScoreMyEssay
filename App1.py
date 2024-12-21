import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict
import nltk
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

### Neural Network Section ###

# Define features (X) and target (y)
X = df[['char_count', 'word_count', 'sent_count', 'avg_word_len', 
        'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 
        'adv_count', 'readability_score', 'content_diversity']]
y = df['final_score']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### Build Neural Network Model ###

# Define early stopping callback
early_stopping = EarlyStopping(patience=50, min_delta=0.001, restore_best_weights=True)

# Define early stopping callback
early_stopping = EarlyStopping(patience=50, min_delta=0.001, restore_best_weights=True)

# Build neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(11,)),  # Input layer (11 features)
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1, activation='linear')  # Output layer (single regression value)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

### Train the Model ###

# Train the model with early stopping
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping]
)

### Evaluate the Model ###

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Test MAE: {test_mae:.4f}")

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

def score_essay():
    print("Enter the essay text:")
    text = input()
    
    features = extract_features_from_text(text)
    essay_data = pd.DataFrame([features])
    scaled_essay_data = scaler.transform(essay_data)
    
    prediction = model.predict(scaled_essay_data)
    1
    print("Predicted Score: ", prediction[0][0])
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
