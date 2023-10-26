# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from flaml import AutoML

import numpy as np

DataComp1 = pd.read_csv('DataComp1.csv')

DataComp1 = DataComp1.dropna(subset=['Result'])


card_mapping = {'A': 11, 'K': 10, 'Q': 10, 'J': 10, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

for col in ["Player's Card 1", "Player's Card 2", "Dealer's Card 1", "Dealer's Card 2"]:
    DataComp1[col] = DataComp1[col].map(card_mapping)



le = LabelEncoder()
DataComp1["Player's Action 1"] = le.fit_transform(DataComp1["Player's Action 1"])
DataComp1["Dealer's Action 1"] = le.fit_transform(DataComp1["Dealer's Action 1"])


X = DataComp1.drop(columns=["Round", "Result"])
y = DataComp1["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                            learning_rate=0.1, n_estimators=100, max_depth=5),
    
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, 
                                            random_state=42),
    
    'Support Vector Machine': SVC(C=1.0, kernel='rbf', gamma='scale'),
    
    'Logistic Regression': LogisticRegression(C=1.0, solver='liblinear', 
                                              random_state=42)
}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict the outcomes on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")


param_grid = {
    'learning_rate': np.linspace(0.01, 1, 20),
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'n_estimators': [50, 100, 150, 200, 250],
    'subsample': np.linspace(0.5, 1, 10),
    'colsample_bytree': np.linspace(0.5, 1, 10),
    'gamma': np.linspace(0, 1, 10),
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.1, 0.5, 1, 2]
}

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
rs = RandomizedSearchCV(xgb_model, param_grid, n_iter=100, scoring='accuracy', n_jobs=-1, cv=5, random_state=42)
rs.fit(X_train, y_train)

best_xgb = rs.best_estimator_

y_pred = best_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized XGBoost Accuracy: {accuracy * 100:.2f}%")



from flaml import AutoML

automl = AutoML()


settings = {
    "time_budget": 120, 
    "metric": 'accuracy',
    "task": 'classification',
    "estimator_list": ['xgboost'], 
    "log_file_name": 'flaml_log.log' 
}

automl.fit(X_train, y_train, **settings)

xgb_model = automl.model.estimator
xgb_model.get_booster().save_model('flaml_xgboost_model.model')

print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print(f'Best accuracy on validation data: {1-automl.best_loss:.4g}')

y_pred = automl.predict(X_test)
print('Predicted labels:', y_pred)
print('True labels:', y_test)

from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

