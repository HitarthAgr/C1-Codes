import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# 1. Load dataset
data = pd.read_csv("Telecom_Customer_Churn.csv")

# 2. Explore dataset
print(data.head())
print(data.shape)
print(data.info())
print(data.isnull().sum())

# 3. Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# 4. Handle missing values
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

# 5. Remove duplicate records
data = data.drop_duplicates()

# 6. Statistical description
print(data.describe())

# 7. Check outliers using boxplots
plt.boxplot(data["tenure"])
plt.title("Tenure Outliers")
plt.show()

plt.boxplot(data["MonthlyCharges"])
plt.title("Monthly Charges Outliers")
plt.show()

# 8. Visualize relationships
sns.pairplot(data[["tenure", "MonthlyCharges", "TotalCharges"]])
plt.show()

# 9. Split dataset
X = data.drop("Churn", axis=1)
y = data["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Export cleaned dataset
data.to_csv("Cleaned_Telecom_Customer_Churn.csv", index=False)

print("Data cleaning completed!")