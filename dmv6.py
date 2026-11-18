import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("customer_shopping_data.csv")

# 2. Display first five records
print(df.head())

# 3. Count records for each shopping mall
print(df.groupby("shopping_mall").count())

# 4. Total sales for each shopping mall
branch_sales = df.groupby("shopping_mall").sum(numeric_only=True)

# 5. Total sales for each product category
category_sales = df.groupby("category").sum(numeric_only=True)

# 6. Display top-performing malls
print(branch_sales.sort_values(by="price", ascending=False))

# 7. Display top-selling categories
print(category_sales.sort_values(by="price", ascending=False))

# 8. Group sales by mall and category
combined = df.groupby(["shopping_mall", "category"]).sum(numeric_only=True)
print(combined)

# 9. Pie chart for sales by mall
plt.pie(branch_sales["price"], labels=branch_sales.index)
plt.title("Sales by Shopping Mall")
plt.show()

# 10. Pie chart for sales by category
plt.pie(category_sales["price"], labels=category_sales.index)
plt.title("Sales by Product Category")
plt.show()

# 11. Grouped bar chart
pivot = df.pivot_table(
    index="shopping_mall",
    columns="category",
    values="price",
    aggfunc="sum"
)

pivot.plot(kind="bar", figsize=(10, 6))
plt.title("Category-wise Sales by Shopping Mall")
plt.xlabel("Shopping Mall")
plt.ylabel("Total Sales")
plt.show()
