
# Hebrew Handwritten Character Classifier (Keras)

##  Contact Info

- Shlomo Hadar - Shlomoh03@gmail.com
- Matan Tenenboim - tenenmatan@gmail.com

---

## Environment
- Microsoft Windows 10
- Python 3.10

---

## 📌 Overview

This program is a complete pipeline for **classifying Hebrew handwritten characters** using a **Convolutional Neural Network (CNN)** in **Keras**. It includes:

- Data preprocessing (resizing, padding, inverting)
- One-hot encoding of labels
- Training a CNN
- Evaluating accuracy per letter
- Testing different regularization configurations
- Plotting training curves
- Saving model configuration and confusion matrix

---

## 📂 Input Data

The program expects a directory with **27 subfolders**, each named `0` to `26`, where:
- Each folder contains grayscale images of one Hebrew letter.
- The folder index corresponds to a specific Hebrew letter, including final forms.

---

## 🧠 What the Program Does

1. **Preprocesses images**:
   - Converts them to grayscale
   - Pads them to squares
   - Resizes to 32×32
   - Inverts pixel values (for white-on-black effect)
   - Normalizes pixel values to [0, 1]

2. **Encodes labels** using one-hot encoding

3. **Splits data** into training, validation, and test sets

4. **Builds and trains a base CNN model**

5. **Trains 8 models** with various **regularization configurations**, including:
   - No regularization
   - L1 and L2 regularization
   - Dropout
   - Combinations of Dropout + L2

6. For each configuration:
   - Plots **accuracy and loss** over epochs
   - Evaluates **test accuracy**
   - Saves best model's:
     - Architecture (`final_model_configuration.json`)
     - Accuracy per letter (`final_model_train_test_per_letter_accuracy.txt`)
     - Confusion matrix as a plot (optional) and CSV (`confusion_matrix.csv`)
     - Accuracy/Loss plot (`final_model_precision_and_loss_plot.png`)

7. Displays a **summary** of test accuracies for all configurations.

---

## 📤 Outputs

| Output File | Description |
|-------------|-------------|
| `final_model_configuration.json` | JSON file containing the model architecture |
| `confusion_matrix.csv` | Confusion matrix of model predictions vs. true labels |
| `final_model_train_test_per_letter_accuracy.txt` | Accuracy per Hebrew letter and average accuracy |
| `final_model_precision_and_loss_plot.png` | Plot of training and validation accuracy/loss for best model |

---

## 🧪 Evaluation Details

- **Per-letter accuracy** is calculated and written to a text file.
- **Confusion matrix** is computed and saved in CSV format.
- All evaluation steps support **one-hot or label-encoded labels**.

---

## ▶️ Running the Program

Make sure:
- All 27 folders (`0` to `26`) with images are in the same directory as the script.
- You have installed the required packages:
  ```bash
  pip install -r requirement.txt
  ```
- Then simply run:
  ```bash
  python 04_keras_mnist_classification.py
  ```
