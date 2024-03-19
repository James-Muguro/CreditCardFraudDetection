# Importing necessary libraries
import pandas as pd
import numpy as np

# Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Data preprocessing libraries
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression  # for logistic regression
from sklearn.neighbors import KNeighborsClassifier  # for k-nearest neighbors
from sklearn.svm import SVC  # for support vector classifier
from sklearn.tree import DecisionTreeClassifier  # for decision tree classifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier  # ensemble methods
from xgboost import XGBClassifier  # XGBoost classifier

# Metrics libraries
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load the dataset 'creditcard.csv' into a pandas DataFrame
df = pd.read_csv('creditcard.csv')

# Set pandas option to display all columns
pd.set_option('display.max_columns', None)

# Print the first few rows of the DataFrame
df.head()


# Explore the datatype of each column in the DataFrame
df.info()

# Get the shape of the DataFrame (number of rows and columns)
df.shape

# Retrieve the column names of the DataFrame
df.columns

# Import the Normalizer class from sklearn.preprocessing
from sklearn.preprocessing import Normalizer

# Create an instance of the Normalizer class with L2 normalization
normalizer = Normalizer(norm='l2')

# Fit the normalizer to the data and transform it
normalized_data = normalizer.fit_transform(df)

# Print the normalized data
print(normalized_data)

# Display the distribution of legitimate (Class 0) and fraudulent (Class 1) transactions
df['Class'].value_counts()

# Separate the data into two subsets: legitimate transactions and fraudulent transactions
legit = df[df['Class'] == 0]  # Subset containing legitimate transactions (Class 0)
fraud = df[df['Class'] == 1]  # Subset containing fraudulent transactions (Class 1)

# Display statistical measures of the legitimate transactions data
legit.describe()


# Display statistical measures of the 'Amount' column for legitimate transactions
legit['Amount'].describe()


# Display statistical measures of the fraudulent transactions data
fraud.describe()


# Display statistical measures of the 'Amount' column for fraudulent transactions
fraud['Amount'].describe()


# Compare the mean values of each column for both legitimate and fraudulent transactions
df.groupby('Class').mean()


# Under-Sampling: building a sample dataset containing a similar distribution of normal transactions and fraudulent transactions
# Sample 492 random rows from the legitimate transactions DataFrame
legit_sample = legit.sample(n=492)

# Concatenate the sampled legitimate transactions DataFrame with the entire fraudulent transactions DataFrame
new_df = pd.concat([legit_sample, fraud], axis=0)

# Print the first 5 rows of the new dataset
new_df.head()


# Get the distribution of the classes for the subsample dataset
new_df['Class'].value_counts()


# Check for missing values in each column
missing_values = df.isnull().sum().sort_values(ascending=False)

# If there are no missing values, print a message
if missing_values.sum() == 0:
    print("No missing values found.")
else:
    # Otherwise, print the count of missing values for each column
    print("Missing values found:")
    print(missing_values)


# Get the shape of the DataFrame new_df (number of rows and columns)
new_df.shape


# Splitting the data into Features (X) and Targets (y)
X = new_df.drop(columns='Class', axis=1)  # Features (independent variables) excluding the 'Class' column
y = new_df['Class']  # Target variable (dependent variable), 'Class' column



# Splitting the data into Training data and Testing data
# X_train: Features for training
# X_test: Features for testing
# y_train: Target labels for training
# y_test: Target labels for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Check whether the data is split in an 80:20 ratio
print(X.shape, X_train.shape, X_test.shape)



# Call the Random Forest Classifier model
model = RandomForestClassifier(random_state=42)


# Importing necessary libraries
from sklearn.pipeline import Pipeline

# Create a pipeline for the model
pipeline = Pipeline([
    ('model', model)  # 'model' step with the RandomForestClassifier model
])

# Perform cross-validation
scores = cross_val_score(pipeline, X_train, y_train, cv=5)

# Calculate mean accuracy
mean_accuracy = scores.mean()

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("Model:", RandomForestClassifier())
print("Cross-validation Accuracy:", mean_accuracy)
print("Test Accuracy:", accuracy)
print('Recall Score: ', recall)
print('Precision Score: ', precision)
print('F1 Score: ', f1)

# Save the best model
import pickle
pickle.dump(pipeline, open('iris_model.dot', 'wb'))


# Importing necessary libraries
from sklearn.metrics import confusion_matrix

# Define class labels for the confusion matrix
LABELS = ['Normal', 'Fraud'] 

# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred) 

# Plot confusion matrix
plt.figure(figsize=(12, 12)) 
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion Matrix") 
plt.ylabel('True Class') 
plt.xlabel('Predicted Class') 
plt.show()


# Create a copy of the subsample dataset 'new_df' and assign it to 'df_train'
df_train = new_df.copy()

# Display the first few rows of the 'df_train' DataFrame
df_train.head()


def display_feature_importance(model, percentage, top_n=34, plot=False):
    # Get features (X) and target variable (y)
    X = df_train.drop('Class', axis=1)
    y = df_train['Class']
    
    # Fit the model using the features and target variable
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    
    # Sort features by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Calculate the threshold based on a specified percentage of the top feature importance
    threshold = percentage / 100 * feature_importance_df.iloc[0]['Importance']
    
    # Select features that meet the threshold
    selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
    
    # Print selected features
    print("Selected Features by {} \n \n at threshold {}%; {}".format(type(model).__name__, percentage, selected_features))
    
    if plot:
        # Set seaborn color palette to "viridis"
        sns.set(style="whitegrid", palette="viridis")
    
        # Plot the top features
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
        plt.title('Feature Importance for {}'.format(type(model).__name__))
        plt.show()
        
    # Add 'Class' to the list of selected features
    selected_features.append('Class')
        
    return selected_features


# List to store selected features for each model and trial percentage
selected_features_xgb = []

# Initilize AUC List 
auc_scores = []

# List of trial percentages
trial_percentages = [3, 5, 10, 20, 40]

# Loop over each trial percentage
for percentage in trial_percentages:
    # Get selected features for each model
    xgb_selected_features = display_feature_importance(XGBClassifier(random_state=42), percentage=percentage)

    # Append selected features to the respective lists
    selected_features_xgb.append(xgb_selected_features)

    # X and y 
    X = df_train.drop('Class', axis=1)
    y = df_train['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Fit models on training data
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train[[feature for feature in xgb_selected_features if feature != 'Class']], y_train, verbose=0)

    # Predict probabilities on the test set
    xgb_pred_proba = xgb_model.predict_proba(X_test[[feature for feature in xgb_selected_features if feature != 'Class']])[:, 1]

    # Calculate AUC scores and append to the list
    from sklearn.metrics import roc_auc_score

    auc_xgb = roc_auc_score(y_test, xgb_pred_proba)
    auc_scores.append((auc_xgb, percentage))

    # Sorted AUC 
    sorted_auc = sorted(auc_scores, reverse=True)

# Print Each AUC with Percentage 
for score, percentage in sorted_auc:
    print(f'The AUC for {type(XGBClassifier(random_state=42)).__name__} with {percentage}% of top features is {score:.4f}')


# List of important features extracted using XGBoost
imp_fea = ['V14', 'V10', 'V4', 'V7', 'V21', 'V8', 'V20', 'V3', 'V5', 'V11', 'V12', 'V26', 'V17', 'Class']

# Update df_train to include only the important features
df_train = df_train[imp_fea]

# Display the first few rows of the updated DataFrame
df_train.head()


# Check the shape of the DataFrame df_train
df_train.shape


def train_random_forest(data, target):
    """
    Train a Random Forest model using the provided data.

    Parameters:
        data (DataFrame): The input DataFrame containing features and target.
        target (str): The name of the target column.

    Returns:
        best_rf_model (RandomForestClassifier): The trained Random Forest model with the best hyperparameters.
        best_params (dict): The best hyperparameters selected by GridSearchCV.
        accuracy (float): The accuracy of the best model on the test set.
    """
    # Dictionary to store LabelEncoders for each categorical column
    label_encoders = {}

    # split the data into X and y
    X = data.drop(target, axis=1)
    y = data[target]

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Define the Random Forest model
    rf_model = RandomForestClassifier(random_state=0, class_weight='balanced')

    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best hyperparameters
    print('Best Hyperparameters:')
    print(best_params)

    # Train the model on the full training set
    best_rf_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred_rf = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_rf)
    precision = precision_score(y_test, y_pred_rf)
    recall = recall_score(y_test, y_pred_rf)

    print(f'Accuracy on Test Set: {accuracy:.2f}')
    print(f'Precision on Test Set: {precision:.2f}')
    print(f'Recall on Test Set: {recall:.2f}')
    
    # Visualize the confusion matrix
    LABELS = ['Normal', 'Fraud'] 
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred_rf) 
    plt.figure(figsize=(12, 12)) 
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d"); 
    plt.title("Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

    return best_rf_model, best_params, accuracy



train_random_forest(df_train,'Class')


def train_xgb_classifier(data, target):
    """
    Train an XGBoost classifier using the provided data.

    Parameters:
        data (DataFrame): The input DataFrame containing features and target.
        target (str): The name of the target column.

    Returns:
        best_xgb_model (XGBClassifier): The trained XGBoost model with the best hyperparameters.
        best_params (dict): The best hyperparameters selected by GridSearchCV.
    """
    # split the data into X and y
    X = data.drop(target, axis=1)
    y = data[target]

    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Define the XGBClassifier model
    xgb_model = XGBClassifier(random_state=0)

    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 1, 2]
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_xgb_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best hyperparameters
    print('Best Hyperparameters:')
    print(best_params)

    # Train the model on the full training set
    best_xgb_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred_xgb = best_xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_xgb)
    precision = precision_score(y_test, y_pred_xgb)
    recall = recall_score(y_test, y_pred_xgb)

    print(f'Accuracy on Test Set: {accuracy:.2f}')
    print(f'Precision on Test Set: {precision:.2f}')
    print(f'Recall on Test Set: {recall:.2f}')
    
    # Visualize the confusion matrix
    LABELS = ['Normal', 'Fraud'] 
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred_xgb) 
    plt.figure(figsize=(12, 12)) 
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d"); 
    plt.title("Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()

    return best_xgb_model, best_params



train_xgb_classifier(df_train,'Class')


# List of trained models
models = ['XGB Classifier', 'RandomForestClassifier']

# List of accuracy scores for each model
accuracy_scores = [accuracy, accuracy]

# Find the index of the maximum accuracy
best_accuracy_index = accuracy_scores.index(max(accuracy_scores))

# Print the best model with its accuracy
print(f'Best Accuracy: {accuracy_scores[best_accuracy_index]:.2f} with Model: {models[best_accuracy_index]}')
