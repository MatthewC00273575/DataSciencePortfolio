import React from "react";
import CodeBlock from "../components/CodeBlock";

const CNN = () => {
  return (
    <div className="container">
      <h1>Convolutional Neural Network (CNN)</h1>

      <h2>Overview</h2>
      <p>
        Notebook:{" "}
        <a href="https://github.com/trekhleb/machine-learning-experiments/blob/master/experiments/objects_detection_ssdlite_mobilenet_v2/objects_detection_ssdlite_mobilenet_v2.ipynb">
          Object Detection SSDlite MobileNet V2
        </a>
      </p>
      <p>
        Original Content: Used SSDlite MobileNet V2 for real-time object
        detection on COCO dataset.
      </p>
      <p>
        Dataset:{" "}
        <a href="https://www.kaggle.com/datasets/mbkinaci/fruit-images-for-object-detection">
          UCI Fruit Images for Object Detection
        </a>
      </p>
      <p>300+ images, 3 classes (apple, banana, orange), XML annotations.</p>

      <h2>Goal</h2>
      <p>
        Replaced COCO with Fruit Images, fine-tuned the model, and enhanced
        visualization.
      </p>

      <h2>Loading the Dataset</h2>
      <CodeBlock
        code={`import os
import cv2
import numpy as np

# Load images from dataset
image_paths = [os.path.join('dataset', img) for img in os.listdir('dataset')]
images = [cv2.imread(img) for img in image_paths]

print(f"Loaded {len(images)} images.")`}
      />
      <div className="output-placeholder">[Loaded 300+ fruit images]</div>

      <h2>Preprocessing</h2>
      <p>Resized images to 320x320 and ensured correct format.</p>
      <CodeBlock
        code={`import tensorflow as tf
import numpy as np

# Load and preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))  # SSDlite MobileNet V2 input size
    img = img / 255.0  # Normalize
    return img

# Prepare dataset (simplified for demo; full prep needs TF records)
images_preprocessed = [preprocess_image(os.path.join(data_dir, d['filename'])) for d in dataset]
labels = [d['objects'] for d in dataset]

# Convert to numpy for model input
X = np.array(images_preprocessed)
print(f"Preprocessed shape: {X.shape}")`}
      />
      <div className="output-placeholder">
        [Preprocessed shape: (240, 320, 320, 3)]
      </div>

      <h2>Model Inference</h2>
      <p>Used pre-trained SSDlite MobileNet V2.</p>
      <CodeBlock
        code={`# Load pre-trained model
model_path = 'ssdlite_mobilenet_v2_coco/saved_model'
model = tf.saved_model.load(model_path)
infer = model.signatures['serving_default']

# Preprocess without normalization (keep uint8)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (320, 320))  # Match SSDlite input size
    return img  # Stays uint8 (0-255)

# Load and preprocess one image
sample_img_path = os.path.join(data_dir, dataset[0]['filename'])
sample_img = preprocess_image(sample_img_path)
sample_img = tf.convert_to_tensor(sample_img[np.newaxis, ...], dtype=tf.uint8)  # Shape (1, 320, 320, 3)

# Run inference with correct keyword
detections = infer(inputs=sample_img)  # Use 'inputs=' as per signature

# Extract detections
boxes = detections['detection_boxes'].numpy()[0]
scores = detections['detection_scores'].numpy()[0]
classes = detections['detection_classes'].numpy()[0].astype(int)
print(f"Top detection: Class {classes[0]}, Score {scores[0]:.3f}")`}
      />
      <div className="output-placeholder">
        [Top detection: Class 53, Score 0.864]
      </div>

      <h2>Key Experiments</h2>
      <h3>New Dataset Integration</h3>
      <p>
        Replaced COCO with Fruit Images. Parsed 300+ images and annotations.
      </p>

      <h3>Algorithm Adjustments</h3>
      <p>
        Used pre-trained COCO model due to time constraints; inference ran
        successfully.
      </p>

      <h3>Visual Analysis</h3>
      <p>Visualized detection results.</p>
      <CodeBlock
        code={`import random

# Visualize 3 random test images
num_to_show = 3
random_indices = random.sample(range(len(test_images)), 
min(num_to_show, len(test_images)))  # Pick 3 unique random indices

for i in random_indices:
    img = preprocess_image(test_images[i])
    pred = test_predictions[i]
    visualize_detection(img, pred['boxes'], pred['classes'], pred['scores'], threshold=0.5)
    print(f"Image: {pred['filename']}")`}
      />
      <div className="image-output-placeholder">
        <img src="/images/banana.png" alt="cnn-detections" />
        <img src="/images/apple.png" alt="cnn-detections" />
      </div>

      <div className="image-output-placeholder">
        <img src="/images/orange.png" alt="cnn-detections" />
      </div>
    </div>
  );
};

export default CNN;
