import React from "react";
import CodeBlock from "../components/CodeBlock";

const RNN = () => {
  return (
    <div className="container">
      <h1>Recurrent Neural Network (RNN)</h1>

      <h2>Overview</h2>
      <p>
        Notebook:{" "}
        <a href="https://github.com/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb">
          Numbers Summation RNN
        </a>
      </p>
      <p>
        Original Content: Trains an RNN to sum two random numbers using digit
        sequences.
      </p>
      <p>Dataset: Custom CSV of number triplets (a, b, c where c = a + b).</p>

      <h2>Goal</h2>
      <p>
        Replace synthetic pairs with triplet CSV, predict sums, and improve
        accuracy.
      </p>

      <h2>Loading the Dataset</h2>
      <CodeBlock
        code={`import pandas as pd
import numpy as np

# Mock dataset (replace with real CSV if you have one)
data = pd.DataFrame({
    'a': np.random.randint(0, 101, 5000),
    'b': np.random.randint(0, 101, 5000)
})
data['c'] = data['a'] + data['b']
data.to_csv('number_triplets.csv', index=False)

# Load
df = pd.read_csv('number_triplets.csv')
print(df.head())`}
      />
      <div className="output-placeholder">
        [Sample triplet dataset: a, b, c (sum)]
      </div>

      <h2>Preprocessing</h2>
      <p>One-hot encode inputs for sequence training.</p>
      <CodeBlock
        code={`import numpy as np

# Convert numbers to one-hot encoding
def one_hot_encode(numbers, max_digits=6):
    return np.array([[int(digit) for digit in str(num).zfill(max_digits)] for num in numbers])

X = one_hot_encode(data[['a', 'b']])
y = one_hot_encode(data['c'])`}
      />
      <div className="output-placeholder">[Encoded sequences]</div>

      <h2>Training the Model</h2>
      <p>
        Trained RNN with 64 units and tested different activation functions.
      </p>
      <CodeBlock
        code={`# Using Relu
import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(6, vocab_size), return_sequences=False),
    layers.Dense(3 * vocab_size, activation='relu'),  # ReLU
    layers.Reshape((3, vocab_size))
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(X_onehot, y_onehot, epochs=100, batch_size=32, validation_split=0.2, verbose=1) `}
      />
      <div className="output-placeholder">
        [Epoch 100/100 125/125 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy:
        0.9179 - loss: 0.0018 - val_accuracy: 0.8810 - val_loss: 0.0022]
      </div>

      <h2>Key Experiments</h2>
      <h3>New Dataset Integration</h3>
      <p>
        Replaced random pairs with triplet CSV, preprocessed sequences for RNN.
      </p>

      <h3>Algorithm Adjustments</h3>
      <p>
        Scaled SimpleRNN to 64 units for larger data; tested different epochs
        (e.g., 50,100 epochs).
      </p>
      <p>Swapped softmax for ReLU, Sigmoid, Linear, Tanh with MSE loss.</p>

      <h3>Results Summary</h3>
      <p>Original: ~95% accuracy on random pairs.</p>
      <p>New Dataset: ReLU reached ~91% accuracy at 100 epochs.</p>

      <h3>Visual Analysis</h3>
      <CodeBlock
        code={`import matplotlib.pyplot as plt

# Plot loss curve
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()`}
      />
      <div className="image-output-placeholder">
        <img src="/images/rnntraining.png" alt="rnn-loss-curve" />
      </div>
    </div>
  );
};

export default RNN;
