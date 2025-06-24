import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/airfoil_self_noise.csv')

X = df.drop('Sound_pressure', axis=1)
y = df['Sound_pressure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
import numpy as np
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

