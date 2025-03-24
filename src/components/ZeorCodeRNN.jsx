import React, { useState } from 'react';
import CodeBlock from '../components/CodeBlock';

const ZeroCodeRNN = () => {
    const [fullSizeImage, setFullSizeImage] = useState(null);

    const handleImageClick = (imageId) => {
        setFullSizeImage(fullSizeImage === imageId ? null : imageId);
    };

    return (
        <div className="container">
            <h1>Zero Code RNN Model with ChatGPT</h1>
            <h2>Overview</h2>
            <p>Reference: <a href="https://github.com/trekhleb/machine-learning-experiments/blob/master/experiments/numbers_summation_rnn/numbers_summation_rnn.ipynb">Numbers Summation RNN Notebook</a></p>
            <p>Original Content: Implements an RNN to sum two numbers using TensorFlow, with code for data prep, model building, and training.</p>
            <p>Dataset: Custom dataset with three columns (a, b, c) where c = a + b, based on UCI Mushroom dataset sums.</p>
            <p>~8,000 samples, two numeric inputs (a, b), one numeric output (c).</p>

            <h2>Intentions</h2>
            <p>For this assignment, I aimed to create two RNN models to sum two numbers (c = a + b): one I coded myself and one using ChatGPT with no code. My goal with the "zero code" approach was to leverage ChatGPT to design and simulate an RNN model conceptually, based on my dataset, without writing any code myself. I wanted to specify key details like epochs, activation functions, loss, and accuracy, and test the model with sample inputs, inspired by the Numbers Summation RNN notebook.</p>

            <h2>Dataset</h2>
            <p>I used a custom dataset derived from summing pairs of numbers, formatted as a CSV-like file with columns 'a', 'b', and 'c' (no header in the first row). Each row represents two numbers (a, b) and their sum (c), e.g., "35,9,44". It contains ~8,000 rows of numeric data, where c is always the exact sum of a and b. Below is a conceptual loading step (no code executed here—just for context):</p>
            <CodeBlock code={`# Conceptual dataset loading (not executed)
import pandas as pd
url = "my_dataset.txt"
columns = ['a', 'b', 'c']
data = pd.read_csv(url, header=None, names=columns)
print(data.head())
# Expected: Rows like [35, 9, 44], [11, 33, 44], etc.`} />
            <div className="output-placeholder">[Dataset preview: ~8000 rows, 3 cols; e.g., a=35, b=9, c=44]</div>

            <h2>ChatGPT Prompt Process</h2>
            <p>I guided ChatGPT through a series of prompts to build, train, and test the RNN model conceptually. Below are screenshots of my seven prompts and their responses, showing the step-by-step process.</p>

            <h3>Prompt 1</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/Prompt1.png"
                    alt="prompt1"
                    className={fullSizeImage === 'prompt1' ? 'full-size' : ''}
                    onClick={() => handleImageClick('prompt1')}
                />
            </div>

            <h3>Prompt 2</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/prompt2.png"
                    alt="prompt2"
                    className={fullSizeImage === 'prompt2' ? 'full-size' : ''}
                    onClick={() => handleImageClick('prompt2')}
                />
            </div>

            <h3>Prompt 3</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/prompt3(1).png"
                    alt="prompt3(1)"
                    className={fullSizeImage === 'prompt3-1' ? 'full-size' : ''}
                    onClick={() => handleImageClick('prompt3-1')}
                />
            </div>

            <h3>Prompt 4</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/prompt3(2).png"
                    alt="prompt3(2)"
                    className={fullSizeImage === 'prompt3-2' ? 'full-size' : ''}
                    onClick={() => handleImageClick('prompt3-2')}
                />
            </div>

            <h3>Prompt 5</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/prompt4-5.png"
                    alt="prompt4-5"
                    className={fullSizeImage === 'prompt4-5' ? 'full-size' : ''}
                    onClick={() => handleImageClick('prompt4-5')}
                />
            </div>

            <h3>Prompt 6</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/chatgptcode.png"
                    alt="chatgptcode"
                    className={fullSizeImage === 'chatgptcode' ? 'full-size' : ''}
                    onClick={() => handleImageClick('chatgptcode')}
                />
            </div>

            <h3>Prompt 7</h3>
            <div className="image-output-placeholder">
                <img
                    src="/images/chatgptoutput.png"
                    alt="chatgptoutput"
                    className={fullSizeImage === 'chatgptoutput' ? 'full-size' : ''}
                    onClick={() => handleImageClick('chatgptoutput')}
                />
            </div>

            <h2>Key Outcomes</h2>
            <p>Through these prompts, ChatGPT designed an RNN with no code, simulating a model that learned to sum numbers with decreasing loss and high accuracy. Final predictions were close to true sums (e.g., 44 for 35+9), demonstrating the power of conceptual ML design.</p>
        </div>
    );
};

export default ZeroCodeRNN;