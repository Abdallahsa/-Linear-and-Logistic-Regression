import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' (non-interactive)
import pandas as pd


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


# Removing records with missing values
data_without_missing = data.dropna()


# separate the features from targets
features = data_without_missing.drop(['Max_Loan_Amount', 'Loan_Status'], axis=1)
targets = data_without_missing[['Max_Loan_Amount', 'Loan_Status']]


# Convert 'Dependents' column to numeric values and convert the type to int64
features['Dependents'] = features['Dependents'].replace('3+', 3)
features['Dependents'] = features['Dependents'].astype('int64')
print (features.dtypes)

# Define the features to be converted
features_to_convert = ['Gender', 'Married', 'Education', 'Property_Area']

# Map string values to numeric values (0 and 1)
numeric_mapping = {'Male': 0, 'Female': 1, 'Yes': 1, 'No': 0, 'Graduate': 1, 'Not Graduate': 0, 'Urban': 0, 'Rural': 1, 'Semiurban': 2}


# Apply the mapping to the specified features
features[features_to_convert] = features[features_to_convert].replace(numeric_mapping)

print (features.head(50))
print (features.shape)


# Map string values to numeric values (0 and 1)
map_to_convert = {'Y': 1, 'N': 0}
targets['Loan_Status'] = targets['Loan_Status'].replace(map_to_convert)


print (targets.head(15))



def my_shuffle_data(X, Y, seed=None):
    if seed is not None:
        # Ensure seed is an integer
        seed = int(seed)
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X.iloc[idx], Y.iloc[idx]


X_shuffled, Y_shuffled = my_shuffle_data(features, targets)
print (X_shuffled.head(10))
print (X_shuffled.shape)
print (X_shuffled.dtypes)


shuffled_features_data = X_shuffled.drop(['Loan_ID'], axis=1)
Loan_ID = X_shuffled['Loan_ID']
print (shuffled_features_data.shape)
print (Loan_ID.shape)
print (shuffled_features_data.dtypes)


# Check if numerical features have the same scale
shuffled_features_scaled_data = (shuffled_features_data - shuffled_features_data.mean()) / shuffled_features_data.std()
print(shuffled_features_scaled_data.head(10))
print(shuffled_features_scaled_data.shape)


# Add 'Loan_ID' back to the DataFrame
shuffled_features_scaled_data.insert(0, 'Loan_ID', Loan_ID)


print(shuffled_features_scaled_data.head(10))
print(shuffled_features_scaled_data.shape)



def my_train_test_split(X, y, test_size=0.25):                  # 10*513
    mysplit = len(y) - int(len(y) // (1 / test_size))           # 513-int(513//(20)) == 410
    X_train, X_test = X[:mysplit][:], X[mysplit:][:]            # X_train = [0:410][:], X_test[410:end][:]
    y_train, y_test = y[:mysplit][:], y[mysplit:][:]
    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()

X_train, X_test, y_train, y_test = my_train_test_split(shuffled_features_scaled_data, targets, test_size=0.20)

print (X_test.shape)
print (X_train.shape)



