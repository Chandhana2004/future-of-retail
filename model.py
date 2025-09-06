import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv('bigmart_data.csv')

# Fix missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')

# Encode Item_Fat_Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'Low Fat': 0, 'low fat': 0, 'LF': 0,
    'Regular': 1, 'reg': 1
}).infer_objects(copy=False)

# Drop 'Item_Identifier' (not useful for prediction)
df = df.drop(['Item_Identifier'], axis=1)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=[
    'Item_Type', 'Outlet_Identifier',
    'Outlet_Size', 'Outlet_Location_Type',
    'Outlet_Type'
])

# Features and Target
X = df.drop('Item_Outlet_Sales', axis=1)
y = df['Item_Outlet_Sales']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict and evaluate using RMSE
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ Model trained. RMSE:", rmse)

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
