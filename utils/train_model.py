
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor


train = pd.read_csv('./data/df_train.csv')

selected_features = [
    'Seville_humidity',
    'Valencia_temp',
    'Barcelona_temp_max',
]

X_train = train[selected_features]  # Features
y_train = train['load_shortfall_3h']

# Instantiate the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Pickle model for use within our API
save_path = '../assets/trained-models/load_shortfall_simple_lm_regression.pkl'

print (f"Training completed. Saving model to: {save_path}")

with open(save_path, 'wb') as model_file:
    pickle.dump(rf_model, model_file)