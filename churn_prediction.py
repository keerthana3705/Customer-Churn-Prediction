import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

df.columns = df.columns.str.strip()

# Drop unnecessary column
#df.drop("customerID", axis=1, inplace=True)

#Fix TotalCharges column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

#Remove missing values
df.dropna(inplace=True)

y = df["Churn"]
X = df.drop("Churn", axis=1)

#Convert categorical to numeric
#df = pd.get_dummies(df, drop_first=True)

# Convert categorical to numeric
# le = LabelEncoder()
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col] = le.fit_transform(df[col])

le = LabelEncoder()
y = le.fit_transform(y)

X = pd.get_dummies(X, drop_first=True)

# Split data
# X = df.drop("Churn", axis=1)
# y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
# model = RandomForestClassifier()
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# print(df.columns)

# df.columns = df.columns.str.strip()

# y = df["Churn"]
# X = df.drop("Churn", axis=1)

# X = pd.get_dummies(X, drop_first=True)

print("\nClassification Report:\n", classification_report(y_test, y_pred))