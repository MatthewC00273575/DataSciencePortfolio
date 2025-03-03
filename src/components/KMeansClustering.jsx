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
            <p>I load the Wholesale Customers dataset, dropping metadata columns.</p>
            <CodeBlock code={`import pandas as pd

df = pd.read_csv('Wholesale customers data.csv')
X = df.drop(['Channel', 'Region'], axis=1)
df.head()`} />
            <div className="output-placeholder">[Dataset preview image]</div>

            <h2>Feature Scaling</h2>
            <p>I scale the spending features since K-Means relies on distance.</p>
            <CodeBlock code={`from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`} />
            <div className="output-placeholder">[Scaled dataset preview image]</div>

            <h2>Applying K-Means</h2>
            <p>I apply K-Means with an initial k=3.</p>
            <CodeBlock code={`from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
print(f"Silhouette Score: {silhouette_score(X_scaled, labels):.4f}")`} />
            <div className="output-placeholder">[Cluster score output: ~0.4580]</div>

            <h2>Key Experiments</h2>
            <h3>New Dataset Integration</h3>
            <p>I replaced blobs/digits with Wholesale Customers, scaling 6 spending features. Baseline k=3 silhouette was ~0.458.</p>
            <h3>Algorithm Adjustments</h3>
            <p>Tuned k: Best ~0.458 at k=3 after testing 2-10, balancing cluster quality.</p>
            <h3>Visual Analysis</h3>
            <p>2D Clusters: PCA scatter showed distinct groups with some overlap; centroids marked centers.</p>
            <p>Spending Patterns: Bar plot revealed segments like high-fresh vs. balanced spenders.</p>
        </div>
    );
};

export default KMeansClustering;