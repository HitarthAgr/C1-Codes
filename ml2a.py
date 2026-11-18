
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import make_pipeline
import numpy as np

# 1. Load dataset
df = pd.read_csv("uber.csv")

# 2. Preprocessing: remove missing values
df = df.dropna()

# Select useful columns
cols = ['pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude',
        'passenger_count', 'fare_amount']

df = df[cols]

# 3. Remove invalid values and outliers
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

# Remove invalid geographical coordinates
df = df[
    (df['pickup_longitude'].between(-180, 180)) &
    (df['dropoff_longitude'].between(-180, 180)) &
    (df['pickup_latitude'].between(-90, 90)) &
    (df['dropoff_latitude'].between(-90, 90))
]

# 4. Check correlation
sns.heatmap(df.corr(), annot=True)
plt.show()

# 5. Separate input and output
X = df.drop('fare_amount', axis=1)
y = df['fare_amount']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create three regression models
models = {
    "Linear": LinearRegression(),
    "Ridge": make_pipeline(StandardScaler(), Ridge()),
    "Lasso": make_pipeline(StandardScaler(), Lasso())
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(name)
    print("R2 Score:", r2)
    print("RMSE:", rmse)
    print()
