import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor

california_housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
df['MedHouseVal'] = california_housing.target


df2=df.copy()
for col in df2.select_dtypes(include=['object']).columns:
    df2[col]=df2[col].astype('category').cat.codes

plt.figure(figsize=(13, 10))
sns.heatmap(df2.corr(),annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

columns_to_remove = ['AveBedrms', 'Longitude']
df = df.drop(columns=columns_to_remove)


X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 5, 7, 9, 12],
    'subsample': [0.5, 0.7, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9, 1]
}

xgb = XGBRegressor()
random_search = RandomizedSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_iter=20, n_jobs=-1, verbose=3, random_state=42)
random_search.fit(X_train, y_train)

# Train best XGBoost model with early stopping
best_params = random_search.best_params_
xgb_best = XGBRegressor(**best_params)
xgb_best.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)

# Stacking Model
stacked_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(**best_params)),
        ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)),
        ('rf', RandomForestRegressor(n_estimators=200, max_depth=10))
    ],
    final_estimator=XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
)

stacked_model.fit(X_train, y_train)

# Evaluate model
y_pred = stacked_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f" Stacked Model MAE: {mae:.4f}")

# Save trained model
joblib.dump(stacked_model, 'house_price_model.pkl')
print(" Model saved as 'house_price_model.pkl'")
