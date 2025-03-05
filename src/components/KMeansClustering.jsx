import React from 'react';
import CodeBlock from '../components/CodeBlock';

const KMeansClustering = () => {
    return (
        <div className="container">
            <h1>K-Means Clustering</h1>
            <h2>Overview</h2>
            <p>Notebook: <a href="https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-K-Means.ipynb">K-Means Clustering</a></p>
            <p>Original Content: Applies K-Means to synthetic blobs and digits data, with elbow method for k selection.</p>
            <p>Dataset: <a href="https://archive.ics.uci.edu/dataset/292/wholesale+customers">UCI Wholesale Customers Dataset</a></p>
            <p>440 samples, 6 numerical features (annual spending: fresh, milk, grocery, frozen, detergents, deli).</p>

            <h2>Goal</h2>
            <p>I replace blobs/digits with the Wholesale Customers Dataset, cluster spending patterns, and enhance K-Means with tuning and interpretation.</p>

            <h2>Loading the Dataset</h2>
            <p>I load the Wholesale Customers dataset, dropping metadata columns, and display initial info.</p>
            <CodeBlock code={`import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('Wholesale customers data.csv')

# Drop non-spending columns
X = data.drop(['Channel', 'Region'], axis=1)

print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(X.head())`} />
            <div className="output-placeholder">[Dataset preview: 440 rows, 6 features]</div>

            <h2>Feature Scaling</h2>
            <p>I scale the spending features since K-Means relies on distance.</p>
            <CodeBlock code={`from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`} />
            <div className="output-placeholder">[Scaled dataset preview]</div>

            <h2>Applying K-Means</h2>
            <p>I apply K-Means with an initial k=3 and evaluate with silhouette score.</p>
            <CodeBlock code={`from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Baseline K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_

print(f"Silhouette Score (k=3): {silhouette_score(X_scaled, labels):.3f}")`} />
            <div className="output-placeholder">[Silhouette score output: ~0.458]</div>

            <h2>Key Experiments</h2>
            <h3>New Dataset Integration</h3>
            <p>I replaced blobs/digits with Wholesale Customers, scaling 6 spending features. Baseline k=3 silhouette was ~0.458.</p>
            <h3>Algorithm Adjustments</h3>
            <p>Tuned k: Best ~0.458 at k=3 after testing 2-10, balancing cluster quality.</p>
            <CodeBlock code={`import matplotlib.pyplot as plt

# Test k from 2 to 10
inertias = []
sil_scores = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o')
plt.xlabel('k'); plt.ylabel('Inertia')
plt.title('Elbow Method for k Selection')

plt.subplot(1, 2, 2)
plt.plot(k_values, sil_scores, marker='o')
plt.xlabel('k'); plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. k')
plt.tight_layout()
plt.show()

# Best k
best_k = k_values[np.argmax(sil_scores)]
print(f"Best k (by silhouette): {best_k}, Score: {max(sil_scores):.3f}")`} />
            <div className="image-output-placeholder"><img src="/images/elbowcurve-silhscore.png" alt="elbowcurve-silhscore" /></div>
            <h3>Visual Analysis</h3>
            <p>2D Clusters: PCA scatter showed distinct groups with some overlap; centroids marked centers.</p>
            <CodeBlock code={`from sklearn.decomposition import PCA

# Reduce to 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='x', s=200, label='Centroids')
plt.xlabel('PCA Component 1'); plt.ylabel('PCA Component 2')
plt.title(f'K-Means Clusters (k={3})')
plt.legend()
plt.show()

print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")`} />
            <div className="image-output-placeholder"><img src="/images/k-means-clusters.png" alt="k-means-clusters" /></div>
            <p>Spending Patterns: Bar plot revealed segments like high-fresh vs. balanced spenders.</p>
            <CodeBlock code={`# Add labels to original data
data['Cluster'] = labels

# Mean spending per cluster
cluster_means = data.groupby('Cluster').mean()
print(cluster_means)

# Bar plot
cluster_means.plot(kind='bar', figsize=(10, 6))
plt.title('Average Spending by Cluster')
plt.ylabel('Spending (unscaled)')
plt.show()`} />
            <div className="image-output-placeholder"><img src="/images/avg-cluster.png" alt="avg-cluster plot" /></div>

        </div>
    );
};

export default KMeansClustering; 