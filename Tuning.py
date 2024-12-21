import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from App2 import EssayScorer  # Import your EssayScorer class

# Load your dataset (assuming it's in the same format as before)
data = pd.read_csv("Processed_data.csv")

# Define the feature columns and target column
X = data[EssayScorer.MODEL_FEATURES]
y = data['final_score']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the Keras model as a Scikit-learn estimator
def create_keras_model(hidden_layers=1, neurons=64, activation='relu'):
    model = Sequential()
    model.add(Dense(neurons, activation=activation, input_shape=(X.shape[1],)))
    for i in range(1, hidden_layers):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    return model

keras_estimator = KerasRegressor(build_fn=create_keras_model, epochs=200, batch_size=32, verbose=0)

# Define the hyperparameter search space
param_grid = {
    'hidden_layers': [1, 2, 3],  # Number of hidden layers
    'neurons': [32, 64, 128, 256],  # Number of neurons in each hidden layer
    'activation': ['relu', 'tanh', 'selu']  # Activation function for hidden layers
}

# Perform Grid Search with Cross-Validation (5-fold)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(keras_estimator, param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_scaled, y)

# Print the best hyperparameters and the corresponding score
print("Best Hyperparameters:")
print(grid_search.best_params_)
print("Best Score (Negative MSE):", grid_search.best_score_)
print("Best Model's Architecture:")
best_model = grid_search.best_estimator_.model
print(best_model.summary())