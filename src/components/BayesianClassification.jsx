import React from 'react';
import CodeBlock from '../components/CodeBlock';

const BayesianClassification = () => {
    return (
        <div className="container">
            <h1>Bayesian Classification</h1>
            <h2>Overview</h2>
            <p>Notebook: <a href="https://github.com/jakevdp/PythonDataScienceHandbook/blob/8a34a4f653bdbdc01415a94dc20d4e9b97438965/notebooks/05.05-Naive-Bayes.ipynb">Naive Bayes</a></p>
            <p>Original Content: Covers Gaussian and Multinomial Naive Bayes with examples (Iris dataset, text classification with 20 Newsgroups).</p>
            <p>Dataset: <a href="https://archive.ics.uci.edu/dataset/73/mushroom">UCI Mushroom Dataset</a></p>
            <p>8,124 samples, 22 categorical features (e.g., cap-shape, odor), binary label (edible or poisonous).</p>

            <h2>Goal</h2>
            <p>I aim to expand Jake VanderPlas’ Naive Bayes notebook by replacing the 20 Newsgroups dataset with the raw UCI Mushroom Dataset, implementing significant adjustments like feature selection, hyperparameter tuning, and visualizations to enhance the model.</p>

            <h2>Loading the Dataset</h2>
            <p>I load the UCI Mushroom dataset, defining custom column names since it lacks a header, and display initial info.</p>
            <CodeBlock code={`import pandas as pd
import numpy as np

# Load dataset (download 'agaricus-lepiota.data' from UCI)
url = "agaricus-lepiota.data"
columns = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
           'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
           'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
           'stalk-surface-below-ring', 'stalk-color-above-ring',
           'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
           'ring-type', 'spore-print-color', 'population', 'habitat']
data = pd.read_csv(url, header=None, names=columns)

# Quick look
print(data.head())
print(data.info())
print(data['class'].value_counts())`} />
            <div className="output-placeholder">[Dataset preview: 8124 rows, 23 cols; ~4208 edible, ~3916 poisonous]</div>

            <h2>Data Preprocessing</h2>
            <p>I clean missing values ('?' in stalk-root), encode categorical features into binary columns, and split the data for training.</p>
            <CodeBlock code={`# Replace '?' with NaN and drop rows with missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Encode categorical features
X = pd.get_dummies(data.drop('class', axis=1))
y = data['class'].map({'e': 0, 'p': 1})

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`} />
            <div className="output-placeholder">[Processed dataset: ~5624 rows, 98 features]</div>

            <h2>Applying Naive Bayes</h2>
            <p>I train a baseline Multinomial Naive Bayes classifier with alpha=1.0 and evaluate its accuracy.</p>
            <CodeBlock code={`from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Fit baseline model
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))`} />
            <div className="output-placeholder">[Baseline accuracy output: ~0.966]</div>

            <h2>Key Experiments</h2>
            <h3>New Dataset Integration</h3>
            <p>I swapped 20 Newsgroups for Mushrooms, encoding 22 features into 98 binary columns and dropping NaNs. Baseline accuracy reached ~96.6%.</p>
            <h3>Algorithm Adjustments</h3>
            <p>Feature Selection: Used VarianceThreshold (0.01) to reduce dimensionality, maintaining accuracy at ~96.5% with fewer features.</p>
            <CodeBlock code={`from sklearn.feature_selection import VarianceThreshold

# Remove low-variance features
selector = VarianceThreshold(threshold=0.01)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)

# Refit model
model_selected = MultinomialNB(alpha=1.0)
model_selected.fit(X_train_selected, y_train)
y_pred_selected = model_selected.predict(X_test_selected)

print(f"Selected Features Accuracy: {accuracy_score(y_test, y_pred_selected):.3f}")`} />
            <div className="output-placeholder">[Selected features accuracy output: ~0.965]</div>
            <p>Hyperparameter Tuning: Tuned alpha (0.1-2.0), slightly refining performance.</p>
            <h3>Visual Analysis</h3>
            <p>Feature Importance: Bar plot highlighted odor and spore-print-color as top predictors, aligning with mycology.</p>
            <CodeBlock code={`import matplotlib.pyplot as plt

# Feature importance
log_probs = model.feature_log_prob_[1]
feature_names = X_train.columns
importance = np.abs(model.feature_log_prob_[1] - model.feature_log_prob_[0])
top_idx = np.argsort(importance)[-10:]
top_features = feature_names[top_idx]
top_importance = importance[top_idx]

plt.figure(figsize=(10, 6))
plt.barh(top_features, top_importance, color='skyblue')
plt.xlabel('Importance (Abs. Log Prob Difference)')
plt.title('Top 10 Features Influencing Naive Bayes Predictions')
plt.tight_layout()
plt.show()`} />
            <div className="image-output-placeholder"><img src="/images/nb-featureimportance - Copy.png" alt="feature importance" /></div>
            <p>Confusion Matrix: Heatmap showed ~96.6% accuracy, few false negatives, ensuring safety.</p>
            <CodeBlock code={`from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Edible', 'Poisonous'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix for Naive Bayes')
plt.show()

print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")`} />
            <div className="image-output-placeholder"><img src="/images/NBConfusionMatrix.png" alt="NB confusion matrix" /></div>
        </div>
    );
};

export default BayesianClassification; 