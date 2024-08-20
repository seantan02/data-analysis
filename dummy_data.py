import pandas as pd
import random

# Read the CSV file into a DataFrame
df = pd.read_csv('data/data.csv')

# Apply the subtraction based on the condition
df['_value'] = df.apply(
    lambda row: float(row['_value']) - random.randint(1,10) if row['_measurement'] == 'group_duration' else row['_value'],
    axis=1
)

df['_value'] = df.apply(
    lambda row: float(row['_value']) - random.randint(1,100) if row['_measurement'] == 'iteration_duration' else row['_value'],
    axis=1
)

df['_value'] = df.apply(
    lambda row: random.randint(0,1) if row['_measurement'] == 'failed_requests' else row['_value'],
    axis=1
)

df['_value'] = df.apply(
    lambda row: 0 if row['_value'] < 0 else row['_value'],
    axis=1
)

# Save the modified DataFrame to a new CSV file
df.to_csv('data/new2.csv', index=False)

print("Processing complete. Saved as 'new.csv'.")