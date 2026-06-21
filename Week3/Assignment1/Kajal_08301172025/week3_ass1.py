import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Load dataset
df = pd.read_csv("agriculture_yield_dataset.csv")


# ====================================
# Q1. DATASET OVERVIEW
# ====================================

print("Shape of Dataset:", df.shape)
print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])
print("\nColumn Names:")
print(df.columns.tolist())
print("\nFirst 10 Records:")
print(df.head(10))

# ====================================
# Q2. DATA TYPES AND MISSING VALUES
# ====================================

print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
missing_cols = df.columns[df.isnull().sum() > 0]
if len(missing_cols) > 0:
    print("\nColumns with Missing Values:")
    print(missing_cols.tolist())
else:
    print("\nNo Missing Values Found.")

# ====================================
# Q3. DESCRIPTIVE STATISTICS
# ====================================
stats = df.describe()
print("\nSummary Statistics:")
print(stats)

print("\nFeature with Highest Mean:")
print(stats.loc['mean'].idxmax())

print("\nFeature with Highest Standard Deviation:")
print(stats.loc['std'].idxmax())

# ====================================
# Q4. DISTRIBUTION ANALYSIS
# ====================================

numerical_cols = ['rainfall_mm','temperature_c','fertilizer_kg','yield_ton_per_hectare']
for col in numerical_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# ====================================
# Q5. CROP TYPE ANALYSIS
# ====================================

crop_counts = df['crop_type'].value_counts()
print("\nCrop Type Counts:")
print(crop_counts)

print("\nMost Frequent Crop:")
print(crop_counts.idxmax())

plt.figure(figsize=(8,5))
sns.countplot(x='crop_type', data=df)
plt.title('Crop Type Distribution')
plt.xticks(rotation=45)
plt.show()

# ====================================
# Q6. SOIL TYPE ANALYSIS
# ====================================

soil_counts = df['soil_type'].value_counts()
print("Frequency of Each Soil Type:")
print(soil_counts)

print("\nMost Common Soil Type:")
print(soil_counts.idxmax())

plt.figure(figsize=(6,4))
sns.countplot(x='soil_type', data=df)
plt.title('Count Plot of Soil Types')
plt.xlabel('Soil Type')
plt.ylabel('Count')
plt.show()

# ====================================
# Q7. YIELD DISTRIBUTION
# ====================================

plt.figure(figsize=(6,4))
sns.histplot(df['yield_ton_per_hectare'], bins=20, kde=True)
plt.title('Yield Distribution')
plt.xlabel('yield_ton_per_hectare')
plt.ylabel('Frequency')
plt.show()

# ====================================
# Q8. SCATTER PLOT ANALYSIS
# ====================================

plt.figure(figsize=(6,4))
sns.scatterplot(
    x='rainfall_mm',
    y='yield_ton_per_hectare',
    data=df
)
plt.title('Rainfall vs Yield')
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(
    x='fertilizer_kg',
    y='yield_ton_per_hectare',
    data=df
)
plt.title('Fertilizer vs Yield')
plt.show()

print("\nCorrelation with Yield:")
print(df.corr(numeric_only=True)['yield_ton_per_hectare'].sort_values(ascending=False))

# ====================================
# Q9. CORRELATION ANALYSIS
# ====================================

corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Top 3 correlated features with yield
yield_corr = corr_matrix['yield_ton_per_hectare'].drop('yield_ton_per_hectare')
top3 = yield_corr.abs().sort_values(ascending=False).head(3)
print("\nTop 3 Features Correlated with Yield:")
print(top3)

# ====================================
# Q10. GROUP BASED ANALYSIS
# ====================================

crop_avg = df.groupby('crop_type')['yield_ton_per_hectare'].mean()
print("\nAverage Yield by Crop Type:")
print(crop_avg)
print("\nCrop with Highest Average Yield:")
print(crop_avg.idxmax())
soil_avg = df.groupby('soil_type')['yield_ton_per_hectare'].mean()
print("\nAverage Yield by Soil Type:")
print(soil_avg)
print("\nSoil Type with Highest Average Yield:")
print(soil_avg.idxmax())

# ====================================
# Q11. FEATURE ENCODING
# ====================================

# Identify categorical columns
categorical_cols = df.select_dtypes(include='object').columns
print("\nCategorical Columns:")
print(categorical_cols.tolist())

# One-Hot Encoding
encoded_df = pd.get_dummies( df, columns=categorical_cols, drop_first=True)
print("\nFirst 5 Rows After Encoding:")
print(encoded_df.head())
print("\nEncoded Dataset Shape:")
print(encoded_df.shape)

# ====================================
# Q12. FEATURE SELECTION
# ====================================

# Target variable
y = encoded_df['yield_ton_per_hectare']
# Input features
X = encoded_df.drop('yield_ton_per_hectare', axis=1)
print("Target Variable (y): yield_ton_per_hectare")
print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
print("\nFirst 5 rows of X:")
print(X.head())
print("\nFirst 5 rows of y:")
print(y.head())

# ====================================
# Q13. TRAIN-TEST SPLIT
# ====================================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

# ====================================
# Q14. LINEAR REGRESSION MODEL
# ====================================


model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept:")
print(model.intercept_)
# Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nModel Coefficients:")
print(coef_df)
# Feature with highest positive coefficient
highest_feature = coef_df.loc[
    coef_df['Coefficient'].idxmax()
]
print("\nFeature with Highest Positive Coefficient:")
print(highest_feature)