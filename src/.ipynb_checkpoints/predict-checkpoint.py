import joblib
from datetime import datetime

model = joblib.load('../models/inventory_model.pkl')

# Predict for a future date
future_date = '2025-07-20'
day_of_year = datetime.strptime(future_date, '%Y-%m-%d').timetuple().tm_yday
units_pred = model.predict([[day_of_year]])
print(f"Predicted units sold on {future_date}: {units_pred[0]:.0f}")