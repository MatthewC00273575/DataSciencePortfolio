import React from 'react';
import CodeBlock from '../components/CodeBlock';

const BayesianClassification = () => {
    return (
        <div className="container">
            <h1>Bayesian Classification</h1>
            <p>This page walks through the implementation of Naive Bayes classification using the UCI Mushroom dataset.</p>

            <h2>Loading the Dataset</h2>
            <p>We start by loading the UCI Mushroom dataset, replacing the original text dataset used in the reference notebook.</p>
            <CodeBlock code={`import pandas as pd

df = pd.read_csv('mushrooms.csv')
df.head()`} />
            <div className="output-placeholder">[Dataset preview image]</div>

            <h2>Data Preprocessing</h2>
            <p>Since the dataset consists of categorical features, we need to encode them into numerical values for Naive Bayes to process.</p>
            <CodeBlock code={`from sklearn.preprocessing import LabelEncoder

df_encoded = df.apply(LabelEncoder().fit_transform)
df_encoded.head()`} />
            <div className="output-placeholder">[Encoded dataset preview image]</div>

            <h2>Applying Naive Bayes</h2>
            <p>We train a Naive Bayes classifier on the encoded dataset.</p>
            <CodeBlock code={`from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')`} />
            <div className="output-placeholder">[Model accuracy output]</div>
        </div>
    );
};

export default BayesianClassification;
