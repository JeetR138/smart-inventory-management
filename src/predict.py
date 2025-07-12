import joblib
import pandas as pd
from datetime import datetime

# Define product-to-code mapping (must match training encoding)
product_map = {
    "Widget A": 0,
    "Widget B": 1,
    "Widget C": 2
}

# User inputs
future_date = '2025-07-20'           # 🔁 You can change this
product_name = 'Widget B'            # 🔁 You can change this

# Check if product is valid
if product_name not in product_map:
    raise ValueError(f"❌ Product '{product_name}' not found. Choose from: {list(product_map.keys())}")

# Convert date to day of year
day_of_year = datetime.strptime(future_date, '%Y-%m-%d').timetuple().tm_yday
product_code = product_map[product_name]

# Prepare input features as DataFrame
features = pd.DataFrame([[day_of_year, product_code]], columns=['DayOfYear', 'Product_Code'])

# Load trained model
model = joblib.load('../models/inventory_model.pkl')

# Predict
prediction = model.predict(features)

# Display result
print(f"📦 Predicted Units Sold on {future_date} for {product_name}: {int(prediction[0])}")