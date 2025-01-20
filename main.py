import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle

# Load the dataset
try:
    print("Loading dataset...")
    data = pd.read_csv('creditcard.csv')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Overview of the dataset
print(data.head())
print(data.info())

# Check for class imbalance
print(data['Class'].value_counts())

# Visualize the class distribution
sns.countplot(data['Class'])
plt.title('Class Distribution')
plt.show()

print(data.isnull().sum())

# Scale the 'Amount' feature
data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

# Drop irrelevant columns (if any)
data = data.drop(['Time'], axis=1)

# Separate features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
try:
    print("Training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
except Exception as e:
    print(f"Error during model training: {e}")

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Handle class imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Save the model
import joblib

# Save the trained model
joblib.dump(model, 'fraud_detection_model.pkl')
print("Model saved as fraud_detection_model.pkl")



# Load the model to verify
try:
    with open('fraud_detection_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
