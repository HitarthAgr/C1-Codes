import pandas as pd
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("aqi.csv", encoding="latin-1")

# 2. Explore dataset
print(data.head())
print(data.info())
print(data.isnull().sum())

# 3. Select pollutant columns
cols = ["so2", "no2", "rspm", "spm"]

# 4. Handle missing values
for col in cols:
    data[col] = data[col].fillna(data[col].mean())

# 5. Convert date into datetime
data["date"] = pd.to_datetime(
    data["sampling_date"], errors="coerce"
)

data = data.dropna(subset=["date"])

# 6. Create year column
data["year"] = data["date"].dt.year

# 7. Line plot - yearly pollutant trends
yearly = data.groupby("year")[cols].mean()

yearly.plot(kind="line", figsize=(10, 5))
plt.title("Yearly Air Quality Trends")
plt.xlabel("Year")
plt.ylabel("Pollutant Level")
plt.legend()
plt.show()

# 8. Bar plot - average pollutant levels
data[cols].mean().plot(kind="bar")
plt.title("Average Pollutant Levels")
plt.xlabel("Pollutants")
plt.ylabel("Average Level")
plt.show()

# 9. Box plot
data[cols].plot(kind="box")
plt.title("Pollutant Distribution")
plt.show()

# 10. Scatter plot
plt.scatter(data["so2"], data["no2"])
plt.title("SO2 vs NO2")
plt.xlabel("SO2")
plt.ylabel("NO2")
plt.show()

print("Data Visualization Completed!")