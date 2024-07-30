# Install necessary libraries
import os
import sys

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
except ImportError:
    os.system('pip install pandas scikit-learn')
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

# Load the datasets
train_data_path = "C:/Users/ishaa/Downloads/train.csv"
test_data_path = "C:/Users/ishaa/Downloads/test.csv"
submission_path = "C:/Users/ishaa/Downloads/submission.csv"

try:
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
except FileNotFoundError:
    print(f"Error: File not found at the specified path. Please check the file paths.")
    sys.exit(1)

# Display the first few rows of the data to understand its structure
print("Train Data Head:")
print(train_data.head())
print("Test Data Head:")
print(test_data.head())

# Display the column names
print("Train Data Columns:")
print(train_data.columns)
print("Test Data Columns:")
print(test_data.columns)

# Selecting relevant columns
# Update these column names based on your dataset structure
square_footage_col = 'LotArea'
num_bedrooms_bathrooms_col = 'OverallQual'
price_col = 'SalePrice'
id_col = 'Id'

try:
    train_data = train_data[[square_footage_col, num_bedrooms_bathrooms_col, price_col]]
    test_data = test_data[[square_footage_col, num_bedrooms_bathrooms_col, id_col]]
except KeyError as e:
    print(f"Error: Column not found. {e}")
    sys.exit(1)

# Handle missing values (if any)
train_data.dropna(inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Features and target variable
X = train_data[[square_footage_col, num_bedrooms_bathrooms_col]]
y = train_data[price_col]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate the mean squared error
mse = mean_squared_error(y_val, y_val_pred)
print(f"Mean Squared Error: {mse}")

# Make predictions on the test set
test_predictions = model.predict(test_data[[square_footage_col, num_bedrooms_bathrooms_col]])

# Create a DataFrame for the submission
submission = pd.DataFrame({
    'Id': test_data[id_col],
    'SalePrice': test_predictions
})

# Save the submission DataFrame to a CSV file
submission.to_csv(submission_path, index=False)

print("Submission file created successfully at:", submission_path)
