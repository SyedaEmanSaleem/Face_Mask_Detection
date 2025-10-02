# Face Mask Detection using CNN

## 📌 Introduction

Face masks have become an essential protective measure during global pandemics. Detecting whether individuals are wearing masks is crucial in ensuring public health and safety in places such as hospitals, airports, schools, and workplaces.

This project focuses on building a **Convolutional Neural Network (CNN)** model that classifies facial images into two categories:

* **With Mask**
* **Without Mask**

The model has been trained and validated on the publicly available **Face Mask 12k Images Dataset** from Kaggle.

---

## 📂 Dataset Description

The dataset used for this project is the **Face Mask 12k Images Dataset** which contains:

* **Training set:** Images used to fit the CNN model.
* **Validation set:** Images used to tune hyperparameters and monitor overfitting.
* **Test set:** Images kept aside to evaluate the final performance of the model.

Each set contains two classes:

1. **WithMask** – images of people wearing masks.
2. **WithoutMask** – images of people without masks.

The dataset is balanced and contains high-quality labeled images, making it suitable for supervised deep learning tasks.

---

## ⚙️ Methodology

### 1. Data Preprocessing

* All images were resized to a fixed dimension.
* Pixel values were normalized to the range [0,1].
* Data augmentation techniques were applied (rotation, zoom, horizontal flip, etc.) to improve generalization.

### 2. Model Architecture

A custom-built **CNN** was used consisting of:

* **Convolutional layers:** Extracted spatial features from the input images.
* **MaxPooling layers:** Reduced spatial dimensions while retaining important features.
* **Dropout layers:** Prevented overfitting by randomly dropping neurons.
* **Fully connected layers:** Combined extracted features for final classification.
* **Softmax output layer:** Produced probabilities for the two classes.

### 3. Training Setup

* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 30
* **Callbacks Used:**

  * ModelCheckpoint (saved the best model weights)
  * ReduceLROnPlateau (reduced learning rate when validation accuracy plateaued)

---

## 📊 Results

* **Training Accuracy:** ~100%
* **Validation Accuracy:** ~99.6%
* **Test Accuracy:** ~99%

The model showed excellent generalization with very minimal overfitting. The validation accuracy remained consistent across epochs, proving the model’s robustness.

* **Final Evaluation:**

  * Loss: ~0.01
  * Accuracy: ~99%

---

## 🚀 Key Features

* High accuracy (close to 99%) on unseen test data.
* Reliable training and validation performance.
* Well-regularized CNN model with dropout and learning rate scheduling.
* Works effectively on balanced binary classification tasks.

---

## 📈 Conclusion

This project successfully demonstrates the use of **Convolutional Neural Networks (CNNs)** for binary image classification in the domain of **face mask detection**.

The trained model achieved near-perfect accuracy and can be deployed in real-world scenarios to automatically monitor whether individuals are wearing masks. The approach highlights the effectiveness of deep learning in addressing health and safety challenges during pandemics.

---

## 🔮 Future Work

* Integrate the model with **OpenCV** for real-time video detection.
* Deploy on edge devices (e.g., Raspberry Pi) for surveillance applications.
* Extend to multi-class classification (e.g., improper mask wearing).
* Experiment with **Transfer Learning** using pre-trained models like MobileNetV2, ResNet50, and EfficientNet.

---

## 📜 License

This project is open-source and free to use for research and educational purposes.

---
