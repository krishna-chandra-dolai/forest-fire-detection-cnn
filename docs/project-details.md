# Complete Project Details: Forest Fire Detection Using CNN

## 1. Project Title

**Forest Fire Detection Using Convolutional Neural Networks**

This project develops a deep learning based forest fire detection system that classifies images into two categories:

- **Fire**
- **No Fire**

The main model used in the project is **ResNet-50**, a deep Convolutional Neural Network architecture, applied through **transfer learning**.

## 2. Project Identity

| Field | Details |
|---|---|
| Project name | Forest Fire Detection Using Convolutional Neural Networks |
| Domain | Artificial Intelligence, Deep Learning, Computer Vision, Environmental Monitoring |
| Main technique | CNN image classification |
| Core model | ResNet-50 |
| Development language | Python 3.11 |
| Development environment | Jupyter Notebook |
| Frameworks | TensorFlow and Keras |
| Dataset type | Fire and non-fire images |
| Input image size | 224 x 224 pixels |
| Output | Fire / No Fire prediction |
| Report author | Krishna Chandra Dolai |
| Degree context | Master of Computer Applications |
| Guide | Dr. G. Narasimha Rao |
| Institution | Andhra University College of Engineering (A), Andhra University |

## 3. Abstract

Forest fires are dangerous natural disasters that cause ecological damage, loss of biodiversity, property damage, and threats to human life. Traditional fire detection methods such as human watch towers, aerial patrols, and satellite monitoring often suffer from delayed detection, high cost, limited coverage, and false alarms.

This project proposes an image-based forest fire detection system using **Convolutional Neural Networks**, especially the **ResNet-50 architecture**. The system is trained on labelled fire and non-fire images. Images are resized, normalized, augmented, and passed through a fine-tuned ResNet-50 model for binary classification.

The model was trained for **20 epochs** and achieved strong performance:

- **Test accuracy:** 95.80%
- **Validation accuracy:** 96.00%
- **Overall classification accuracy:** 96.00%

The project shows that CNN-based fire detection can support early warning systems and help authorities respond faster to forest fire incidents.

## 4. Problem Statement

Forest fires spread rapidly and can destroy large areas of forest, wildlife habitats, agricultural land, and nearby human settlements. Early detection is critical because even a short delay can allow a small fire to become uncontrollable.

Existing forest fire detection methods face several problems:

- Human observation has limited coverage and depends on visibility and human alertness.
- Ground patrols are slow and expensive.
- Aerial surveillance gives better coverage but has high operational cost.
- Satellite systems may have delay because images are captured at intervals.
- Traditional systems may fail in cloudy, smoky, or remote environments.
- Existing machine learning systems require good-quality datasets and careful preprocessing.

The project solves this by using an automated CNN-based image classification system that can detect fire from images more quickly and accurately.

## 5. Objectives

The main objective is to build a reliable forest fire detection system using deep learning.

Specific objectives:

- Develop a CNN model that classifies images as fire or no fire.
- Use transfer learning with ResNet-50 to improve accuracy and reduce training time.
- Train the model on labelled fire and non-fire image datasets.
- Preprocess images by resizing, normalization, and augmentation.
- Evaluate the model using accuracy, precision, recall, F1-score, loss, and confusion matrix.
- Test the model on unseen images and real-world-like conditions.
- Design the system so it can be integrated with real-time monitoring systems such as cameras, drones, or alert systems.

## 6. Existing System

The report discusses several existing fire detection approaches.

### 6.1 Human Observation and Lookout Towers

Trained personnel watch forest regions from towers or checkpoints.

Limitations:

- Limited visibility.
- Human fatigue.
- Not suitable for huge forest areas.
- Weather can reduce effectiveness.
- Detection may be delayed.

### 6.2 Ground-Based Patrols

Forest officials or emergency teams monitor regions manually.

Limitations:

- Time-consuming.
- Requires manpower.
- Expensive for large areas.
- Cannot continuously cover remote forests.

### 6.3 Satellite-Based Detection

Satellite imagery is used to identify smoke, heat, or fire spots.

Limitations:

- Delay between satellite passes.
- Lower resolution in some cases.
- Cloud cover can reduce visibility.
- Data processing may take time.

### 6.4 Aerial Detection

Aircraft or drones can monitor forest areas from above.

Limitations:

- High equipment and maintenance cost.
- Limited flight time.
- Weather dependent.
- Requires trained operators.

### 6.5 Machine Learning Systems

ML and deep learning systems can detect fire patterns automatically.

Limitations:

- Need labelled datasets.
- Data quality strongly affects accuracy.
- Models may fail on unfamiliar conditions if not trained well.

## 7. Proposed System

The proposed system uses an image-based deep learning pipeline:

1. **Data collection**
2. **Data preprocessing**
3. **Data augmentation**
4. **Model training using ResNet-50**
5. **Model testing**
6. **Fire / no fire prediction**
7. **Alert generation**

The system takes an image as input and predicts whether the image contains forest fire. In a real-time version, the input could come from surveillance cameras, drones, or satellite imagery.

## 8. System Workflow

### Step 1: Data Collection

The dataset contains labelled images in two classes:

- Fire
- No Fire

The report mentions that the dataset was sourced from Kaggle and online repositories.

### Step 2: Preprocessing

Each image is prepared before training:

- Resize image to **224 x 224 pixels**
- Convert to JPG format
- Normalize pixel values to the range 0 to 1
- Ensure correct class label

### Step 3: Data Augmentation

Augmentation is used to increase dataset variety and reduce overfitting.

Techniques used:

- Rotation
- Width shift
- Height shift
- Shear
- Zoom
- Horizontal flip
- Cropping or similar variation methods

### Step 4: Model Training

The model is trained using ResNet-50 with transfer learning. The pre-trained ResNet-50 model has already learned general image features from ImageNet. The project customizes its final layers for binary fire detection.

### Step 5: Prediction

The trained model receives an image and predicts:

- **Fire**, if fire-like visual patterns are detected.
- **No Fire**, if the image does not contain fire.

### Step 6: Alert System

In a deployment scenario, the prediction can be connected to an alert module that notifies forest departments or emergency teams.

## 9. Dataset Details

The report and presentation describe the dataset as follows:

| Dataset property | Details |
|---|---|
| Source | Kaggle / online repositories |
| Classes | Fire and No Fire |
| Training images | 999 |
| Testing or validation images | 999 |
| Image size | 224 x 224 pixels |
| Image format | JPG |
| Dataset organization | Separate folders for train and test sets |

The presentation lists these paths:

```text
Training Dataset Path:
C:\Users\user\Desktop\project\archiv(1)\FireDataset-V6-JPG-Reshaped224\train

Test Dataset Path:
C:\Users\user\Desktop\project\archive(1)\FireDataset-V6-JPG-Reshaped224\test
```

## 10. Technologies Used

### 10.1 Python

Python is used as the main programming language because it has strong support for machine learning, deep learning, data analysis, and image processing.

### 10.2 Jupyter Notebook

Jupyter Notebook is used for writing code, testing preprocessing steps, training the model, visualizing results, and documenting the workflow.

### 10.3 TensorFlow

TensorFlow is used as the core deep learning framework. It supports neural network training, GPU acceleration, and deployment workflows.

### 10.4 Keras

Keras provides a high-level API for building and training the ResNet-50 based CNN model.

### 10.5 NumPy

NumPy is used for numerical operations and array handling.

### 10.6 Pandas

Pandas is used for dataset organization and metadata handling.

### 10.7 Matplotlib and Seaborn

These libraries are used for visualization, such as accuracy/loss curves, confusion matrix, and performance charts.

### 10.8 OpenCV

OpenCV can be used for image reading, resizing, and preprocessing.

## 11. Model Architecture

The project uses **ResNet-50**, a 50-layer residual neural network.

### Why ResNet-50?

ResNet-50 is suitable because:

- It is strong for image classification.
- It uses residual connections to avoid the vanishing gradient problem.
- It can train deep networks more effectively.
- It is available as a pre-trained model in Keras.
- It performs well with transfer learning.

### Architecture Flow

1. Input image: 224 x 224 x 3
2. ResNet-50 base model with ImageNet weights
3. Convolutional feature extraction
4. Residual blocks
5. Global Average Pooling layer
6. Dense layer with ReLU activation
7. Output Dense layer with sigmoid activation
8. Binary output: fire or no fire

### Transfer Learning Setup

The initial ResNet-50 layers are frozen so they retain general image recognition features. The final layers are replaced and trained on the fire detection dataset.

Typical structure:

```python
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(1, activation="sigmoid")(x)
```

## 12. Training Details

| Training item | Details |
|---|---|
| Model | ResNet-50 with transfer learning |
| Input size | 224 x 224 x 3 |
| Batch size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss function | Binary cross-entropy |
| Output activation | Sigmoid |
| Classification type | Binary classification |

Training progress described in the report:

- Epochs 1-4: Accuracy improved from early low values to strong learning.
- Epochs 5-10: Model continued improving and validation performance increased.
- Epochs 11-15: Training accuracy reached very high values and validation accuracy improved.
- Epochs 16-20: Model reached final strong performance with high accuracy and low loss.

By epoch 20:

- Training accuracy reached about **99.70%**.
- Validation/test performance reached around **95.80% to 96.00%**.

## 13. Testing Details

The testing phase checks whether the trained model performs well on unseen data.

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Loss

The report includes these test cases:

| Test case | Purpose | Status |
|---|---|---|
| TC_01 Model Load Test | Check trained model loads correctly | Passed |
| TC_02 Fire Image Classification | Check model detects fire image | Passed |
| TC_03 Non-Fire Image Classification | Check model detects no-fire image | Passed |
| TC_04 Generalization Test | Test complete unseen dataset | Passed |
| TC_05 Rotation Test | Check rotated fire images | Passed |
| TC_06 Blurred Image Test | Check slightly blurred fire images | Passed |
| TC_07 Lighting Condition Test | Check varying lighting conditions | Passed |
| TC_08 Outlier Image Test | Check unrelated images | Passed |
| TC_09 Adversarial Image Test | Check manipulated fire-like image | Passed |
| TC_10 Large Batch Performance Test | Check 100+ image batch processing | Passed |
| TC_11 Scalability Test | Check stream or larger data handling | Passed |
| TC_12 Post-Deployment Test | Check live-source-like images | Passed |

## 14. Results

The project achieved strong performance.

| Metric | Value |
|---|---|
| Test accuracy | 95.80% |
| Validation accuracy | 96.00% |
| Validation loss | 0.0458 |
| Test loss | 0.1589 |
| Overall classification accuracy | 96.00% |

### Classification Report Summary

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Fire | 99.00% | 95.00% | 97.00% |
| No Fire | 87.00% | 97.00% | 92.00% |
| Macro average | 93.00% | 96.00% | 95.00% |
| Weighted average | 96.00% | 96.00% | 96.00% |

### Result Interpretation

The model is very strong at detecting fire images. Its fire precision of 99% means that when the model predicts fire, it is usually correct. The fire recall of 95% means it detects most actual fire images but may still miss a small number of fire cases.

For no-fire images, recall is high at 97%, meaning most non-fire images are correctly identified. Precision is lower at 87%, meaning there are some false alarms where non-fire images are classified as fire.

In real-world fire detection, this behavior is usually acceptable because missing a fire can be more dangerous than generating a small number of false alerts.

## 15. Advantages

- Provides faster fire detection than manual observation.
- Reduces dependence on expensive aerial and ground-based monitoring.
- Uses transfer learning, so training is faster and more efficient.
- Achieves high accuracy on fire/no-fire image classification.
- Can be extended to real-time camera or drone monitoring.
- Data augmentation improves model robustness.
- CNNs automatically learn visual fire features such as flame color, smoke texture, brightness, and fire shapes.

## 16. Limitations

- Dataset size is relatively small for full real-world deployment.
- Performance depends on image quality.
- Extreme smoke, fog, night scenes, or occlusion may affect accuracy.
- Some non-fire images may be falsely classified as fire.
- Real-time deployment needs suitable hardware.
- More diverse datasets are needed for better generalization.
- The project report focuses on image classification, not full fire spread prediction.

## 17. Future Scope

The report identifies several future improvements:

- Expand the dataset with more fire scenarios, vegetation types, weather conditions, and lighting conditions.
- Add satellite imagery, drone images, and thermal images.
- Combine image data with weather data such as temperature, humidity, wind speed, and rainfall.
- Test newer architectures such as EfficientNet, DenseNet, and Vision Transformers.
- Optimize the model for edge devices such as drones, IoT cameras, and field sensors.
- Use model compression methods such as pruning and quantization.
- Integrate the model with fire department alert systems.
- Add explainability methods such as Grad-CAM or SHAP to show which image regions influenced the prediction.
- Develop adaptive learning so the system improves with new data over time.
- Work with forestry experts and environmental scientists to improve dataset quality and deployment reliability.

## 18. Real-Time Deployment Idea

A practical version of this project can work like this:

1. Camera or drone captures forest image.
2. Image is resized to 224 x 224 pixels.
3. Image is normalized.
4. Trained ResNet-50 model predicts fire/no fire.
5. If fire probability is above threshold, system triggers alert.
6. Alert is sent to forest officials with image, location, and timestamp.

Possible deployment components:

- CCTV camera or drone camera
- Edge device or cloud server
- Trained `.h5` or SavedModel model
- Prediction API
- Alert dashboard
- SMS/email/notification system

## 19. Deliverables

The project deliverables can be described as:

- Project report PDF
- Presentation PDF
- Dataset of fire and no-fire images
- Trained CNN/ResNet-50 model
- Jupyter Notebook implementation
- Testing results
- Confusion matrix and classification report
- Documentation for future deployment
- Prototype design for real-time fire alert system

## 20. Viva / Interview Explanation

Short explanation:

> My project is Forest Fire Detection Using CNN. It detects whether an image contains forest fire or not. I used the ResNet-50 deep learning architecture with transfer learning. The dataset contains fire and no-fire images, resized to 224 x 224 pixels. I used TensorFlow and Keras in Python. Data augmentation was applied to improve generalization. The model was trained for 20 epochs and achieved about 95.80% test accuracy and 96.00% validation accuracy. This system can be extended for real-time forest monitoring using cameras, drones, or satellite images.

Technical explanation:

> The system follows an image classification pipeline. First, fire and non-fire images are collected and labelled. Then the images are resized, normalized, and augmented. A pre-trained ResNet-50 model is loaded using ImageNet weights. Its initial layers are frozen, and new dense layers are added for binary classification. The model is trained using the Adam optimizer and binary cross-entropy loss. During testing, the model is evaluated using accuracy, precision, recall, F1-score, and confusion matrix. The final model achieved strong performance and is suitable as a prototype for early forest fire detection.

## 21. Related Archive Note

The folder also contains a related archive named `Predict-the-Forest-Fires.7z`. That archive appears to be a different project about predicting **burned forest area** using the UCI `forestfires.csv` dataset with meteorological variables such as temperature, humidity, wind, rain, FFMC, DMC, DC, ISI, month, and day.

That archive is not the same as the CNN image-classification project in the main PDF. The main PDF project is about detecting fire/no-fire from images using ResNet-50.

## 22. Final Conclusion

The project successfully demonstrates how CNNs, especially ResNet-50 with transfer learning, can be used for forest fire detection from images. The system improves over traditional fire detection methods by offering faster, automated, and more accurate classification. With a reported test accuracy of 95.80% and validation accuracy of 96.00%, the model is a strong prototype for real-time forest fire monitoring.

Future improvements should focus on larger datasets, real-time camera integration, thermal/satellite imagery, weather data integration, model explainability, and deployment on drones or edge devices.
