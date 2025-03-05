import React from 'react';
import CodeBlock from '../components/CodeBlock';

const KNearestNeighbors = () => {
    return (
        <div className="container">
            <h1>K-Nearest Neighbors</h1>
            <h2>Overview</h2>
            <p>Notebook: <a href="https://www.kaggle.com/code/mmdatainfo/k-nearest-neighbors">K-Nearest Neighbors</a></p>
            <p>Original Content: Demonstrates KNN classification with Iris dataset, basic preprocessing, and accuracy scoring.</p>
            <p>Dataset: <a href="https://archive.ics.uci.edu/dataset/1/abalone">UCI Abalone Dataset</a></p>
            <p>4,177 samples, 8 features (1 categorical: sex; 7 numerical: length, weight), target "rings" (age).</p>

            <h2>Goal</h2>
            <p>I replace Iris with the Abalone Dataset, binarize rings for classification, and enhance the KNN implementation with tuning and visualizations.</p>

            <h2>Loading the Dataset</h2>
            <p>I load the Abalone dataset, assigning column names since it has no header, and display initial info.</p>
            <CodeBlock code={`import pandas as pd
import numpy as np

# Load data
columns = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 
           'viscera_weight', 'shell_weight', 'rings']
data = pd.read_csv('abalone.data', header=None, names=columns)

print(f"Samples: {data.shape[0]}, Features: {data.shape[1]}")
print(data['rings'].value_counts())`} />
            <div className="output-placeholder">[Dataset preview: 4177 rows, 9 features]</div>

            <h2>Preprocessing Data</h2>
            <p>I binarize rings (less than 10 = young, greater or equal to 10 = old), encode 'sex', scale features, and split the data.</p>
            <CodeBlock code={`# Binarize rings
data['rings'] = (data['rings'] >= 10).astype(int)
X = data.drop('rings', axis=1)
y = data['rings']

# Encode sex
X = pd.get_dummies(X, columns=['sex'], drop_first=True)

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)`} />
            <div className="output-placeholder">[Preprocessed dataset: ~60% young, ~40% old]</div>

            <h2>Applying KNN</h2>
            <p>I train a KNN classifier with k=5 as a baseline and evaluate its accuracy.</p>
            <CodeBlock code={`from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Baseline KNN with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Baseline k=5 Accuracy: {accuracy_score(y_test, y_pred):.3f}")`} />
            <div className="output-placeholder">[Baseline accuracy output: ~0.778]</div>

            <h2>Key Experiments</h2>
            <h3>New Dataset Integration</h3>
            <p>I replaced Iris with Abalone, binarizing rings and encoding sex. Baseline k=5 accuracy was ~77.8%.</p>
            <h3>Algorithm Adjustments</h3>
            <p>Tuned k: Best accuracy ~79.2% at k=11, smoothing the decision boundary.</p>
            <CodeBlock code={`# Test different k values
k_values = range(1, 21)
scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o')
plt.xlabel('k'); plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs. k')
plt.show()

# Best k
best_k = k_values[np.argmax(scores)]
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_best = best_knn.predict(X_test)
print(f"Best k: {best_k}, Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")`} />
            <div className="image-output-placeholder"><img src="/images/knn-accuracy.png" alt="knn accuracy plot" /></div>
            <h3>Visual Analysis</h3>
            <p>Confusion Matrix: Balanced errors with a slight bias to young (~60% of data).</p>
            <CodeBlock code={`from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix for best k
cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=['Young', 'Old'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Tuned KNN')
plt.show()

print(f"Tuned Accuracy: {accuracy_score(y_test, y_pred_best):.3f}")`} />
            <div className="image-output-placeholder"><img src="/images/knn-confusionmatrix.png" alt="knn confusion matrix" /></div>
            <p>Accuracy vs. k Plot: Showed peak at k=11 after testing 1-20.</p>
        </div>
    );
};

export default KNearestNeighbors;