
import pandas as pd
import matplotlib.pyplot as plt

# Load data from different file formats
df = pd.read_csv("sales.csv")

# Display dataset information
print(df.head())
print(df.info())

# Handle missing values
df = df.dropna()

# Remove duplicate records
df = df.drop_duplicates()

# Calculate total sales
print("Total Sales:", df["Sales_Amount"].sum())

# Group sales by category
category_sales = df.groupby("Product_Category")["Sales_Amount"].sum()
print(category_sales)

# Visualize sales
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.show()
