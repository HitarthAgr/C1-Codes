import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Bengaluru_House_Data.csv")

# 2. Explore dataset
print(df.head())
print(df.shape)
print(df.isnull().sum())

# 3. Remove unnecessary columns
df = df.drop(
    ["area_type", "society", "balcony", "availability"],
    axis=1
)

# 4. Remove missing values
df = df.dropna()

# 5. Extract BHK from size
df["bhk"] = df["size"].apply(
    lambda x: int(x.split(" ")[0])
)

# 6. Convert total_sqft into numeric format
def convert_sqft(x):
    try:
        if "-" in x:
            a, b = x.split("-")
            return (float(a) + float(b)) / 2
        return float(x)
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna()

# 7. Calculate price per square foot
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]

# 8. Group by location
avg_price = df.groupby("location")["price"].mean()

print("-------- Avg Price --------")
print(avg_price.head())

# 9. Identify outliers using boxplot
plt.boxplot(df["price"])
plt.title("House Price Outliers")
plt.show()

# 10. Save cleaned dataset
df.to_csv("Cleaned_RealEstate.csv", index=False)

print("Data Wrangling Completed!")