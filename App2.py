import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import cmudict, wordnet
from nltk import pos_tag, download
from spellchecker import SpellChecker
import nltk

# Download required NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('cmudict')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class EssayScorer:
    MODEL_FEATURES = ['char_count', 'word_count', 'sent_count', 'avg_word_len', 
                       'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 
                       'adv_count', 'readability_score', 'content_diversity']
    
    def __init__(self, model_path="Processed_data.csv"):
        self.model_path = model_path
        self.spell = SpellChecker()
        self.model, self.scaler = self._load_model()

    def _load_model(self):
        # Load dataset and ensure required columns exist
        data = pd.read_csv(self.model_path)
        required_columns = ['char_count', 'word_count', 'sent_count', 'avg_word_len', 
                             'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 
                             'adv_count', 'final_score']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns in the dataset.")
        
        # Feature Engineering
        data['readability_score'] = data.apply(lambda row: self.calculate_readability(row['avg_word_len'], row['word_count'], row['sent_count']), axis=1)
        data['content_diversity'] = data.apply(lambda row: self.calculate_content_diversity(row['noun_count'], row['adj_count'], row['verb_count'], row['adv_count'], row['word_count']), axis=1)

        # Neural Network Section
        X = data[self.MODEL_FEATURES]
        y = data['final_score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build Neural Network Model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(self.MODEL_FEATURES),)),  
            Dense(32, activation='relu'),  
            Dense(1, activation='linear')  
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Train the Model
        early_stopping = EarlyStopping(patience=50, min_delta=0.001, restore_best_weights=True)
        model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test), callbacks=[early_stopping])

        return model, scaler

    @staticmethod
    def calculate_readability(avg_word_len, word_count, sent_count):
        if word_count == 0 or sent_count == 0:
            return 0
        return 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (avg_word_len / word_count)

    @staticmethod
    def calculate_content_diversity(noun_count, adj_count, verb_count, adv_count, total_word_count):
        if total_word_count == 0:
            return 0
        unique_parts_of_speech = noun_count + adj_count + verb_count + adv_count
        return unique_parts_of_speech / total_word_count

    def extract_features_from_text(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Input text must be a non-empty string.")
        
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
        misspelled = self.spell.unknown(words)
        spell_err_count = len(misspelled)
        
               # Calculate parts of speech counts (simplified)
        pos_tags = pos_tag(words)
        noun_count = sum(1 for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS'])
        adj_count = sum(1 for word, pos in pos_tags if pos in ['JJ', 'JJR', 'JJS'])
        verb_count = sum(1 for word, pos in pos_tags if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        adv_count = sum(1 for word, pos in pos_tags if pos in ['RB', 'RBR', 'RBS'])

        # Calculate readability score and content diversity
        readability_score = self.calculate_readability(avg_word_len, word_count, sent_count) if word_count != 0 and sent_count != 0 else 0
        content_diversity = self.calculate_content_diversity(noun_count, adj_count, verb_count, adv_count, word_count) if word_count != 0 else 0

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
            'content_diversity': content_diversity,
            'misspelled_words': list(misspelled)
        }

    def suggest_spelling_corrections(self, features):
        corrections = {}
        for word in features['misspelled_words']:
            corrections[word] = self.spell.correction(word)
            print(f"Misspelled Word: {word}, Suggested Correction: {corrections[word]}")
        return corrections

    def suggest_vocabulary_enhancements(self, text):
        enhancements = {}
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        for word, pos in pos_tags:
            if pos in ['JJ', 'JJR', 'JJS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:  
                synonyms = set()
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.add(lemma.name())
                synonyms = list(synonyms - {word})
                if synonyms:
                    enhancements[word] = synonyms[:3]  
                    print(f"Word: {word}, Suggested Enhancements: {enhancements[word]}")
        return enhancements

    def score_essay(self, text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            print("Error: Input text must be a non-empty string.")
            return
        
        try:
            features = self.extract_features_from_text(text)
            model_features = {key: value for key, value in features.items() if key in self.MODEL_FEATURES}
            essay_data = pd.DataFrame([model_features])
            scaled_essay_data = self.scaler.transform(essay_data)
            
            prediction = self.model.predict(scaled_essay_data)
            
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
            
            print("\nSpelling Corrections:")
            self.suggest_spelling_corrections(features)
            
            print("\nVocabulary Enhancements:")
            self.suggest_vocabulary_enhancements(text)
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def handle_errors(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    return wrapper

@handle_errors
def main():
    scorer = EssayScorer()
    while True:
        print("\nOptions:")
        print("1. Score an Essay")
        print("2. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            print("Enter the essay text:")
            text = input()
            scorer.score_essay(text)
            cont = input("\nWould you like to score another essay? (y/n): ")
            if cont.lower() != 'y':
                break
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()
