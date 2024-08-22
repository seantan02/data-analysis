import numpy as np
import pandas as pd
import datetime

dates = pd.date_range(start="2022-01-01", periods=8, freq='3ME')
X = np.array(dates).reshape(-1, 1)

print(dates)
# Extract year and quarter (3-month intervals)
X_features = pd.DataFrame({
    'year': dates.year,
    'quarter': dates.quarter
})
print(X_features)
# Combine year and quarter as a single feature
X_features['year_quarter'] = X_features['year'] * 4 + X_features['quarter']

# Convert to numerical values for the regression model
X_numeric = X_features['year_quarter'].values.reshape(-1, 1)

print(X_numeric)

x = datetime.datetime(2022, 1, 1)
print(x.year)