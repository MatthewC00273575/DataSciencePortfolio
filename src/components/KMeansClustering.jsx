import React from 'react';
import CodeBlock from '../components/CodeBlock';

const KMeansClustering = () => {
    return (
        <div className="container">
            <h1>K-Means Clustering</h1>
            <p>We explore K-Means clustering using the UCI Wholesale Customers dataset to identify spending patterns.</p>

            <h2>Loading the Dataset</h2>
            <p>We begin by loading the Wholesale Customers dataset, which contains annual spending in various categories.</p>
            <CodeBlock code={`import pandas as pd

df = pd.read_csv('wholesale_customers.csv')
df.head()`} />
            <div className="output-placeholder">[Dataset preview image]</div>

            <h2>Feature Scaling</h2>
            <p>K-Means is distance-based, so we scale features to standardize their values.</p>
            <CodeBlock code={`from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.iloc[:, 2:])`} />
            <div className="output-placeholder">[Scaled dataset preview image]</div>

            <h2>Applying K-Means</h2>
            <p>We apply K-Means clustering with an initial choice of k=3.</p>
            <CodeBlock code={`from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_scaled)
labels = kmeans.labels_`} />
            <div className="output-placeholder">[Cluster labels output]</div>

            <h2>Finding Optimal k</h2>
            <p>We use the elbow method and silhouette score to determine the best number of clusters.</p>
            <CodeBlock code={`import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, km.labels_))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')

plt.show()`} />
            <div className="output-placeholder">[Elbow method and silhouette score plots]</div>
        </div>
    );
};

export default KMeansClustering;
