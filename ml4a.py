
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 1. Load Iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

# 2. Select numerical features
X = data.iloc[:, 0:4]

# 3. Check missing values
print(data.isnull().sum())

# 4. Elbow Method
wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

# Plot elbow graph
plt.plot(range(1, 11), wcss, marker='o')
plt.axvline(x=3, color='red', linestyle='--', label='K = 3')
plt.legend()
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# 5. Apply K-Means with 3 clusters
km = KMeans(n_clusters=3, init='k-means++',
            random_state=42, n_init=10)

clusters = km.fit_predict(X)

# 6. Add cluster labels
data["Cluster"] = clusters

print(data.head())

# 7. Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            color='red', marker='X', s=150)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
