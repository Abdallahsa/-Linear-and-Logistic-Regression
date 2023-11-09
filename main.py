import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt  # Importing pyplot from Matplotlib
from sklearn.model_selection import train_test_split
# Load the dataset
file_path = "loan_old.csv"  # Make sure to provide the correct file path
data = pd.read_csv(file_path)

# Check for missing values
missing_values = data.isnull().sum()
print("Missing values:")
print(missing_values)

# Check the data types of features
data_types = data.dtypes
print("\nData types:")
print(data_types)

# Check if numerical features have the same scale
numerical_data = data.select_dtypes(include=['int64', 'float64'])
scaled_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
print("\nScaled numerical data:")
print(scaled_data)
print("************************************************")

# Creating a pair plot using the scaled numerical data
sns.pairplot(scaled_data.dropna())  # Use the scaled data for the pair plot and drop rows with NaN values
plt.savefig('pairplot.png')  # Save the plot as an image

# Removing records with missing values
data_without_missing = data.dropna()

# Display the number of missing values after removing records
print("\nNumber of missing values after removing records:")
print (data_without_missing)
print ("***************************")
print(data_without_missing.isnull().sum())
print ("***************************")



# Separate features and target columns
features = data_without_missing.drop(['Max_Loan_Amount', 'Loan_Status'], axis=1)
targets = data_without_missing[['Max_Loan_Amount', 'Loan_Status']]

# Display the first few rows of the features and targets datasets
print("\nFeatures:")
print(features.head())

print("\nTargets:")
print(targets.head())


# Split the data into training and testing sets
# Use 80% of the data for training and 20% for testing
# Set a random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set - Features:", X_train.shape, "Targets:", y_train.shape)
print("Testing set - Features:", X_test.shape, "Targets:", y_test.shape)