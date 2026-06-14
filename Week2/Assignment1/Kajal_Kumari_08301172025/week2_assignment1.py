import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Q1. Load the dataset and display first 5 records
df = pd.read_csv("Dataset 2.csv")
print("First 5 Records:")
print(df.head())


# Q2. Number of rows and columns
print("\nShape of Dataset:")
print(df.shape)
print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])


# Q3. Display all column names
print("\nColumn Names:")
print(df.columns.tolist())


# Q4. Identify numerical and categorical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print("\nNumerical Features:")
print(numerical_features)
print("\nCategorical Features:")
print(categorical_features)


# Q5. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())


# Q6. Average age of users
avg_age = df['Age'].mean()
print("\nAverage Age of Users:")
print(round(avg_age, 2))


#  Q7 
avg_watch_hours = df["WatchHoursPerWeek"].mean()
print("Average Watch Hours Per Week:", round(avg_watch_hours, 2))

#  Q8 
avg_monthly_spend = df["MonthlySpend"].mean()
print("Average Monthly Spend:", round(avg_monthly_spend, 2))

#  Q9 
print("\nUsers in Each Subscription Category:")
print(df["SubscriptionType"].value_counts())

#  Q10 
renewed_percent = (df["SubscriptionRenewed"] == "Yes").mean() * 100
print("\nPercentage of Users Who Renewed Subscription:")
print(round(renewed_percent, 2), "%")

#  Q11 
# Convert categorical features to numerical form
le = LabelEncoder()
df_encoded = df.copy()
for col in ["Gender", "SubscriptionType",
            "FavoriteGenre", "SubscriptionRenewed"]:
    df_encoded[col] = le.fit_transform(df_encoded[col])
print("\nEncoded Dataset:")
print(df_encoded.head())


#  Q12 
X = df_encoded.drop("SubscriptionRenewed", axis=1)
y = df_encoded["SubscriptionRenewed"]

print("\nFeature Set (X):")
print(X.columns)

print("\nTarget Variable (y):")
print("SubscriptionRenewed")

#  Q13 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

#  Q14 
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

#  Q15 
y_pred_dt = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:")
print(round(dt_accuracy * 100, 2), "%")

#  Q16 
cm = confusion_matrix(y_test, y_pred_dt)

print("\nConfusion Matrix:")
print(cm)


#  Q17 
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("\nKNN Accuracy:")
print(round(knn_accuracy * 100, 2), "%")

#  Q18 
print("\nComparison of Models")
print("Decision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")
print("KNN Accuracy:", round(knn_accuracy * 100, 2), "%")

if dt_accuracy > knn_accuracy:
    print("Decision Tree performs better.")
elif knn_accuracy > dt_accuracy:
    print("KNN performs better.")
else:
    print("Both models have the same accuracy.")

#  Q19 
X_reg = df_encoded.drop("MonthlySpend", axis=1)
y_reg = df_encoded["MonthlySpend"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg,
    test_size=0.2,
    random_state=42
)

lr_model = LinearRegression()
lr_model.fit(Xr_train, yr_train)

print("\nLinear Regression Model Trained Successfully")

#  Q20 
new_user = [[
    2000,   # UserID
    30,     # Age
    1,      # Gender
    2,      # SubscriptionType
    35,     # WatchHoursPerWeek
    3,      # DevicesUsed
    1,      # FavoriteGenre
    10,     # AdClicks
    1       # SubscriptionRenewed
]]

predicted_spend = lr_model.predict(new_user)

print("\nPredicted Monthly Spending:")
print(round(predicted_spend[0], 2))