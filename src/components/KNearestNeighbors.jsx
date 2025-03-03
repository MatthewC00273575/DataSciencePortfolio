import React from 'react';
import CodeBlock from '../components/CodeBlock';

const KNearestNeighbors = () => {
    return (
        <div className="container">
            <h1>K-Nearest Neighbors</h1>
            <p>We implement KNN classification using the UCI Abalone dataset to predict abalone age (young or old).</p>

            <h2>Loading the Dataset</h2>
            <p>We start by loading the Abalone dataset, which contains physical measurements and ring counts.</p>
            <CodeBlock code={`import pandas as pd

df = pd.read_csv('abalone.data', header=None, 
                 names=['sex', 'length', 'diameter', 'height', 'whole_weight', 
                        'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])
df.head()`} />
            <div className="output-placeholder">[Dataset preview image]</div>

            <h2>Preprocessing Data</h2>
            <p>We binarize the 'rings' column into young (less than 10) and old (greater than or equal to 10), encode 'sex', and scale features for KNN.</p>
            <CodeBlock code={`from sklearn.preprocessing import StandardScaler

# Binarize rings
df['rings'] = (df['rings'] >= 10).astype(int)

# Encode categorical 'sex'
df_encoded = pd.get_dummies(df, columns=['sex'], drop_first=True)

# Features and target
X = df_encoded.drop('rings', axis=1)
y = df_encoded['rings']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)`} />
            <div className="output-placeholder">[Preprocessed dataset preview image]</div>

            <h2>Applying KNN</h2>
            <p>We train a K-Nearest Neighbors classifier with k=5.</p>
            <CodeBlock code={`from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')`} />
            <div className="output-placeholder">[Model accuracy output]</div>

            <h2>Confusion Matrix</h2>
            <p>We visualize classification performance with a confusion matrix.</p>
            <CodeBlock code={`from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['Young', 'Old'])
plt.show()`} />
            <div className="output-placeholder">[Confusion matrix plot]</div>
        </div>
    );
};

export default KNearestNeighbors;