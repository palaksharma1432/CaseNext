#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
import warnings
import sys

warnings.filterwarnings('ignore')

# --- 1. Load Data and Define Features ---
try:
    import openpyxl

    df = pd.read_excel(r"C:\Users\dell\Downloads\Indian_Court_Cases_Dataset_Updated.xlsx") # Replace with your file path
    print("Data loaded successfully.")

    df['Priority_Label'] = df['Priority_Label'].astype(str).str.strip()
    df.dropna(subset=['Priority_Label'], inplace=True)
    print(f"Data contains {len(df)} cases after cleaning.\n")

except (FileNotFoundError, ImportError) as e:
    print(f"Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading or cleaning the file: {e}")
    sys.exit(1)

FEATURE_DEFINITIONS = {
    'Court_Name': {'type': 'categorical'},
    'Case_Type': {'type': 'categorical'},
    'Urgency_Tag': {'type': 'categorical'},
    'Advocate_Names': {'type': 'categorical'},
    'Legal_Sections': {'type': 'categorical'},
    'Past_History': {'type': 'categorical'},
    'Estimated_Impact': {'type': 'categorical'},
    'Media_Coverage': {'type': 'categorical'},
}

TARGET_COLUMN = 'Priority_Label'

# --- Helper Functions ---
def get_feature_lists(definitions, df):
    categorical_features = [name for name, details in definitions.items()
                            if details['type'] == 'categorical' and name in df.columns]
    return categorical_features

def create_preprocessor(categorical_features):
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[('cat', categorical_transformer, categorical_features)],
        remainder='drop'
    )
    return preprocessor

def train_models(X_train, y_train, preprocessor):
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    lr_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'))
    ])

    print("\nTraining models...")
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)
    print("Training complete.")
    return rf_model, lr_model

def get_user_input(definitions, df):
    user_data = {}
    for feature, details in definitions.items():
        if feature not in df.columns:
            user_data[feature] = input(f"Enter value for {feature}: ")
            continue
        options = df[feature].dropna().unique()
        if options.size > 0:
            print(f"\nFeature: {feature}\nOptions:")
            for i, opt in enumerate(options[:20]):
                print(f"{i+1}. {opt[:50]}")
            if len(options) > 20: print("...")
            while True:
                choice = input(f"Enter choice number (1-{len(options)}), 's' to skip, or type custom value: ").strip()
                if choice.lower() == 's':
                    user_data[feature] = np.nan; break
                try:
                    choice_index = int(choice) -1
                    if 0 <= choice_index < len(options):
                        user_data[feature] = options[choice_index]; break
                    print(f"Invalid choice. Enter 1-{len(options)}.")
                except ValueError: print("Invalid input. Try again.")
        else: user_data[feature] = input(f"Enter value for {feature}: ")
    return pd.DataFrame([user_data])


def predict_priority(input_df, rf_model, lr_model):
    try:
        print("\n--- Prediction Results ---")
        
        rf_prediction = rf_model.predict(input_df)[0]
        rf_proba = rf_model.predict_proba(input_df)[0]
        rf_proba_dict = {class_label: f"{prob:.1%}" for class_label, prob in zip(rf_model.classes_, rf_proba)}
        
        lr_prediction = lr_model.predict(input_df)[0]
        lr_proba = lr_model.predict_proba(input_df)[0]
        lr_proba_dict = {class_label: f"{prob:.1%}" for class_label, prob in zip(lr_model.classes_, lr_proba)}

        averaged_probabilities = {}
        for label in rf_model.classes_:
            rf_prob = rf_proba_dict.get(label, 0.0)
            lr_prob = lr_proba_dict.get(label, 0.0)
            try:
                averaged_prob = (float(rf_prob.strip('%')) + float(lr_prob.strip('%')))/2
                averaged_probabilities[label] = f"{averaged_prob:.1f}%"
            except (ValueError, TypeError):
                averaged_probabilities[label] = "N/A"


        print(f"\nRandom Forest Prediction: {rf_prediction}")
        print("Random Forest Prediction Probabilities:")
        for label, prob in sorted(rf_proba_dict.items(), key=lambda x: float(x[1].rstrip('%')), reverse=True):
            print(f"  {label}: {prob}")
        
        print(f"\nLogistic Regression Prediction: {lr_prediction}")
        print("Logistic Regression Prediction Probabilities:")
        for label, prob in sorted(lr_proba_dict.items(), key=lambda x: float(x[1].rstrip('%')), reverse=True):
            print(f"  {label}: {prob}")

        print(f"\nAveraged Prediction Probabilities:")
        for label, prob in sorted(averaged_probabilities.items(), key=lambda x: float(x[1].rstrip('%')) if x[1] != "N/A" else -1, reverse=True):
            print(f"  {label}: {prob}")

        final_prediction = max(averaged_probabilities, key=lambda k: float(averaged_probabilities[k].strip('%')) if averaged_probabilities[k] != "N/A" else -1)
        print(f"\nFinal Prediction (Averaged): {final_prediction}")

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("Please check your input values and try again.")


# --- 2. Preprocessing and Model Training ---
X = df[list(FEATURE_DEFINITIONS.keys())]
y = df[TARGET_COLUMN]

categorical_features = get_feature_lists(FEATURE_DEFINITIONS, df)
preprocessor = create_preprocessor(categorical_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf_model, lr_model = train_models(X_train, y_train, preprocessor)

# --- Evaluation and Prediction ---
print("\n--- Model Evaluation ---")
print(f"Evaluating on {len(X_test)} test cases")

print("\nRandom Forest Performance:")
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")

print("\nLogistic Regression Performance:")
y_pred_lr = lr_model.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")

# --- 3. Get User Input and Predict ---
while True:
    try:
        user_input_df = get_user_input(FEATURE_DEFINITIONS, df)
        predict_priority(user_input_df, rf_model, lr_model)
        
        another = input("\nMake another prediction? (y/n): ").strip().lower()
        if another != 'y':
            break
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please try again with different inputs.")

print("\nScript finished successfully.")


# In[ ]:




