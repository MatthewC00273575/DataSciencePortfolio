import React from 'react';
import CodeBlock from '../components/CodeBlock';

const SupportVectorMachines = () => {
    return (
        <div className="container">
            <h1>Support Vector Machines</h1>
            <p>We implement SVM using the UCI Wine Quality dataset to classify wine as good or bad.</p>

            <h2>Loading the Dataset</h2>
            <p>We start by loading and merging red and white wine datasets.</p>
            <CodeBlock code={`import pandas as pd

df_red = pd.read_csv('winequality-red.csv', sep=';')
df_white = pd.read_csv('winequality-white.csv', sep=';')
df = pd.concat([df_red, df_white], ignore_index=True)
df.head()`} />
            <div className="output-placeholder">[Dataset preview image]</div>

            <h2>Preprocessing Data</h2>
            <p>We binarize the quality column, defining wines with quality =6 as good and the rest as bad.</p>
            <CodeBlock code={`df['quality'] = (df['quality'] >= 6).astype(int)`} />
            <div className="output-placeholder">[Binarized dataset preview image]</div>

            <h2>Applying SVM</h2>
            <p>We train a Support Vector Classifier with a linear kernel.</p>
            <CodeBlock code={`from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df.drop('quality', axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')`} />
            <div className="output-placeholder">[Model accuracy output]</div>

            <h2>Confusion Matrix</h2>
            <p>We visualize classification errors using a confusion matrix.</p>
            <CodeBlock code={`from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()`} />
            <div className="output-placeholder">[Confusion matrix plot]</div>
        </div>
    );
};

export default SupportVectorMachines;
