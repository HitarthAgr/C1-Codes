import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['Target'] = wine.target
X = df.drop('Target', axis=1)
y = df['Target']
X = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y.values
print("Explained Variance:", pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
for label in pca_df['Target'].unique():
    data = pca_df[pca_df['Target'] == label]
    plt.scatter(data['PC1'], data['PC2'], label=label)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.legend()
plt.show()