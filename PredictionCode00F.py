#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from scipy.sparse import save_npz

# Load train JSON data into a DataFrame
with open('train.json', 'r') as train_file:
    train_data = json.load(train_file)

# Create a DataFrame from the loaded train JSON data
train_df = pd.DataFrame(train_data)

# Load test JSON data into a DataFrame
with open('test.json', 'r') as test_file:
    test_data = json.load(test_file)

# Create a DataFrame from the loaded test JSON data
test_df = pd.DataFrame(test_data)

# Handling text columns in train set
text_columns = ['title', 'editor', 'publisher', 'author', 'abstract']  # Specify the text columns

# Preprocessing for text data in train set
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

for col in text_columns:
    train_df[col] = train_df[col].fillna('')  # Fill NaN values with empty string
    train_df[col] = train_df[col].apply(lambda x: x if isinstance(x, str) else str(x))  # Convert non-string values to string
    train_df[col] = train_df[col].apply(lambda x: x.lower())  # Convert text to lowercase
    train_df[col] = train_df[col].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))  # Tokenization and remove stopwords
    train_df[col] = train_df[col].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation
    train_df[col] = train_df[col].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)]))  # Apply stemming

    # Calculate word count for each text column
    train_df[f'word_count_{col}'] = train_df[col].apply(lambda x: len(x.split()))

# Handling text columns in test set (same preprocessing steps)
for col in text_columns:
    test_df[col] = test_df[col].fillna('')  # Fill NaN values with empty string
    test_df[col] = test_df[col].apply(lambda x: x if isinstance(x, str) else str(x))  # Convert non-string values to string
    test_df[col] = test_df[col].apply(lambda x: x.lower())  # Convert text to lowercase
    test_df[col] = test_df[col].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))  # Tokenization and remove stopwords
    test_df[col] = test_df[col].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation
    test_df[col] = test_df[col].apply(lambda x: ' '.join([ps.stem(word) for word in word_tokenize(x)]))  # Apply stemming

    # Calculate word count for each text column
    test_df[f'word_count_{col}'] = test_df[col].apply(lambda x: len(x.split()))

# Replace 'None' with 0 for all columns in train and test sets
train_df.replace('None', 0, inplace=True)
test_df.replace('None', 0, inplace=True)

# Drop duplicates based on 'title' column in train set

train_df.drop_duplicates(subset=['title'], keep='first', inplace=True)

# Output the modified DataFrames to new JSON files
train_df.to_json('preprocessed_train.json', orient='records', indent=2)
test_df.to_json('preprocessed_test.json', orient='records', indent=2)

# Load the preprocessed train data
train_df = pd.read_json('preprocessed_train.json')

# Load the preprocessed test data
test_df = pd.read_json('preprocessed_test.json')

# Combine text columns into a single column for train and test sets
train_df['combined_text'] = train_df['title'] + ' ' + train_df['editor'] + ' ' + train_df['publisher'] + ' ' + train_df['author'] + ' ' + train_df['abstract']
test_df['combined_text'] = test_df['title'] + ' ' + test_df['editor'] + ' ' + test_df['publisher'] + ' ' + test_df['author'] + ' ' + test_df['abstract']

# Initialize CountVectorizer
count_vectorizer = CountVectorizer()

# Fit and transform the combined text data for train set
vectorized_train_data = count_vectorizer.fit_transform(train_df['combined_text'])

# Fit and transform the combined text data for test set using the vocabulary learned from the train set
vectorized_test_data = count_vectorizer.transform(test_df['combined_text'])

# Save the vectorized train and test data in compressed sparse row (CSR) format directly to files
save_npz('vectorized_train_data.npz', vectorized_train_data)
save_npz('vectorized_test_data.npz', vectorized_test_data)

# Save the vocabulary (feature names) to a JSON file
with open('vocabulary.json', 'w') as vocab_file:
    json.dump(count_vectorizer.vocabulary_, vocab_file)

# Define X_train, y_train (assuming 'year' is the target variable)
X_train = vectorized_train_data
y_train = train_df['year']

# Use Lasso for feature selection
lasso = Lasso(alpha=0.1)  # You can adjust the alpha value
lasso.fit(X_train, y_train)

# Select features based on Lasso regularization
model = SelectFromModel(lasso, prefit=True)
X_train_selected = model.transform(X_train)
# Apply the same transformation to your test set
X_test_selected = model.transform(vectorized_test_data)


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


# Load the preprocessed train and test data
train_df = pd.read_json('preprocessed_train.json')
test_df = pd.read_json('preprocessed_test.json')

# Combine text columns into a single column for train and test sets
train_df['combined_text'] = train_df['title'] + ' ' + train_df['editor'] + ' ' + train_df['publisher'] + ' ' + train_df['author'] + ' ' + train_df['abstract']
test_df['combined_text'] = test_df['title'] + ' ' + test_df['editor'] + ' ' + test_df['publisher'] + ' ' + test_df['author'] + ' ' + test_df['abstract']

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the combined text data for train set
vectorized_train_data = tfidf_vectorizer.fit_transform(train_df['combined_text'])

# Fit and transform the combined text data for test set using the vocabulary learned from the train set
vectorized_test_data = tfidf_vectorizer.transform(test_df['combined_text'])

# Save the vectorized train and test data in compressed sparse row (CSR) format directly to files
save_npz('tfidf_vectorized_train_data.npz', vectorized_train_data)
save_npz('tfidf_vectorized_test_data.npz', vectorized_test_data)

# Save the vocabulary (feature names) to a JSON file
with open('tfidf_vocabulary.json', 'w') as vocab_file:
    json.dump(tfidf_vectorizer.vocabulary_, vocab_file)


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from scipy.sparse import load_npz

# Load preprocessed train and test data
train_df = pd.read_json('preprocessed_train.json')
test_df = pd.read_json('preprocessed_test.json')

# Load vectorized train and test data
vectorized_train_data = load_npz('tfidf_vectorized_train_data.npz')
vectorized_test_data = load_npz('tfidf_vectorized_test_data.npz')


# Define the target variable and features
X_train = vectorized_train_data
y_train = train_df['year']

# Initialize regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regression": DecisionTreeRegressor(),
    "Random Forest Regression": RandomForestRegressor(n_estimators=100),
    "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100),
    "Support Vector Regression": SVR()
}

mae_results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)  # Predict on the train set for demonstration
    mae = mean_absolute_error(y_train, predictions)
    mae_results[name] = mae
    print(f"Mean Absolute Error for {name}: {mae}")

# Choose the best model based on MAE
best_model = min(mae_results, key=mae_results.get)
print(f"Best Model based on MAE: {best_model}")

# Use the best model to predict 'year' values in the test set
best_model = models[best_model]
predictions_test = best_model.predict(vectorized_test_data)

# Create output file with predicted 'year' values for the test set
output_data = [{'year': int(pred)} for pred in predictions_test]  # Assuming predictions are numeric
with open('predicted1.json', 'w') as file:
    json.dump(output_data, file, indent=2)


# In[1]:


get_ipython().system('pip install xgboost')


# In[7]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from scipy.sparse import load_npz

# Load preprocessed train and test data
train_df = pd.read_json('preprocessed_train.json')
test_df = pd.read_json('preprocessed_test.json')

# Load vectorized train and test data
vectorized_train_data = load_npz('tfidf_vectorized_train_data.npz')
vectorized_test_data = load_npz('tfidf_vectorized_test_data.npz')

# Define the target variable and features for training
X_train = vectorized_train_data
y_train = train_df['year']

# Initialize XGBoost Regressor
xgb = XGBRegressor()

# Set up hyperparameters for Grid Search
param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.05, 0.1, 0.3],
    'max_depth': [3, 5, 7]
}

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best estimator from Grid Search
best_xgb = grid_search.best_estimator_

# Print the best mean absolute error
print("Best MAE:", -grid_search.best_score_)  # Negate the score to get positive MAE value

# Predict 'year' values in the test set using the best model
predictions_test = best_xgb.predict(vectorized_test_data)

# Create output file with predicted 'year' values for the test set
output_data = [{'year': int(pred)} for pred in predictions_test]  # Assuming predictions are numeric
with open('predictedXGB.json', 'w') as file:
    json.dump(output_data, file, indent=2)



# In[8]:


best_xgb = grid_search.best_estimator_

# Print the best mean absolute error and best parameters
print("Best MAE:", -grid_search.best_score_)  # Negate the score to get positive MAE value
print("Best Parameters:", grid_search.best_params_)


# In[9]:


get_ipython().system('pip install lightgbm')


# In[10]:


import pandas as pd
import json
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

# Load preprocessed train and test data
train_df = pd.read_json('preprocessed_train.json')
test_df = pd.read_json('preprocessed_test.json')

# Load vectorized train and test data (assuming you have already vectorized the data using TF-IDF)
from scipy.sparse import load_npz
vectorized_train_data = load_npz('tfidf_vectorized_train_data.npz')
vectorized_test_data = load_npz('tfidf_vectorized_test_data.npz')

# Define the target variable and features
X_train = vectorized_train_data
y_train = train_df['year']

# Define the LightGBM regressor
lgbm = LGBMRegressor()

# Define hyperparameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    # Add other hyperparameters to tune
}

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and its parameters
best_lgbm = grid_search.best_estimator_
best_params = grid_search.best_params_

# Train the best model on the entire training set
best_lgbm.fit(X_train, y_train)

# Use the best model to predict 'year' values in the test set
predictions_test = best_lgbm.predict(vectorized_test_data)

# Create output file with predicted 'year' values for the test set
output_data = [{'year': int(pred)} for pred in predictions_test]  # Assuming predictions are numeric
with open('predictedLGBM.json', 'w') as file:
    json.dump(output_data, file, indent=2)

# Print the best MAE obtained from GridSearchCV
best_mae = -grid_search.best_score_
print(f"Best MAE: {best_mae}")
print(f"Best parameters: {best_params}")



# In[11]:


# Print the best MAE obtained from GridSearchCV
best_mae = -grid_search.best_score_
print(f"Best MAE: {best_mae}")
print(f"Best parameters: {best_params}")


# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Load preprocessed train and test data
train_df = pd.read_json('preprocessed_train.json')
test_df = pd.read_json('preprocessed_test.json')

# Load vectorized train and test data
vectorized_train_data = load_npz('tfidf_vectorized_train_data.npz')
vectorized_test_data = load_npz('tfidf_vectorized_test_data.npz')

# Define the target variable and features for training
X_train = vectorized_train_data
y_train = train_df['year']

# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Model training - XGBoost with best parameters
xgb_model = xgb.XGBRegressor(learning_rate=0.3, max_depth=7, n_estimators=500)
xgb_model.fit(X_train, y_train)

# Model training - LightGBM with best parameters
lgb_model = lgb.LGBMRegressor(learning_rate=0.2, n_estimators=300)
lgb_model.fit(X_train, y_train)

# Predictions
xgb_preds = xgb_model.predict(X_valid)
lgb_preds = lgb_model.predict(X_valid)

# Combining predictions (simple average)
combined_preds = (xgb_preds + lgb_preds) / 2.0

# Calculate MAE for each model
xgb_mae = mean_absolute_error(y_valid, xgb_preds)
lgb_mae = mean_absolute_error(y_valid, lgb_preds)
combined_mae = mean_absolute_error(y_valid, combined_preds)

print(f"XGBoost MAE: {xgb_mae}")
print(f"LightGBM MAE: {lgb_mae}")
print(f"Combined Model (Average) MAE: {combined_mae}")

# Use the combined model for making final predictions on the test data
xgb_test_preds = xgb_model.predict(vectorized_test_data)
lgb_test_preds = lgb_model.predict(vectorized_test_data)
combined_test_preds = (xgb_test_preds + lgb_test_preds) / 2.0

# Save predictions to a JSON file (assuming predicted_test contains the final predictions)
output_data = [{'year': int(pred)} for pred in combined_test_preds]  # Assuming predictions are numeric
with open('predicted_test combo.json', 'w') as file:
    json.dump(output_data, file, indent=2)


# In[13]:


print(f"XGBoost MAE: {xgb_mae}")
print(f"LightGBM MAE: {lgb_mae}")
print(f"Combined Model (Average) MAE: {combined_mae}")


# In[ ]:




