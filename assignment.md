# Assignment 5: Vision Transformers (100 Points)

**Due Date:** Sunday, 04/07, 11:59 PM EST

In this assignment, we will explore **Vision Transformers (ViTs)** and apply them to an image classification task using the **CIFAR-100 dataset** (callback to our first assignment!), which can be downloaded from the provided link.

As you know, CIFAR-100 consists of **60,000 color images (32x32)** in **100 classes**, with **600 images per class**. There are **50,000 training images** and **10,000 test images**.

Please carefully review the provided readings, especially:
- The foundational Vision Transformer paper
- The annotated transformer implementation from Harvard NLP

You will perform the following tasks:

---

## [5 pts] Part 1: Dataset Preparation
- Download the CIFAR-100 dataset, unzip, and organize it into training, validation, and testing splits.
  - Use **10% of the training data** for validation.

---

## [10 pts] Part 2: Data Preprocessing
- Write preprocessing scripts to resize images appropriately for input to the Vision Transformer model (e.g., **resize to 224×224**).
- Implement **data augmentation techniques** suitable for Vision Transformer training.

---

## [5 pts] Part 3: Setup the Vision Transformer Model (ViT)
- Initialize a Vision Transformer (ViT) model for image classification.
  - You may use a recommended implementation that aligns with the readings or construct one from scratch using PyTorch.
- Clearly **document the model architecture** and **your initialization strategy**.

---

## [10 pts] Part 4: Training Procedure
- Write a training procedure to train the Vision Transformer model using the training and validation sets.
- Clearly document **hyperparameters** (learning rate, batch size, epochs, etc.) and justify your choices.
- **Do not use pre-trained weights.**

---

## [10 pts] Part 5: Evaluation Procedure
- Write scripts to evaluate your trained Vision Transformer model on the test set.
- Calculate and report **overall classification accuracy**.

---

## [40 pts] Part 6: Metrics and Analysis
- Report evaluation metrics suitable for multiclass classification:
  - Accuracy  
  - Confusion Matrix  
  - F1 Score  
  - Precision & Recall  
  - ROC-AUC Score (one-vs-rest approach)
- Discuss results and analyze model performance across different classes, highlighting strengths and weaknesses.
- This should be a **formal LaTeX report**, similar to previous assignments (e.g., Assignment 1).

---

## [15 pts] Part 7: Documentation
- Write a clear **README** file explaining how to run your code, including:
  - Installation requirements (dependencies, environment setup)
  - Step-by-step instructions for training and evaluation
  - Explanation of scripts and their functions

---

## [5 pts] Part 8: Training Logs
- Provide **detailed training logs** showing accuracy, loss, and validation progress for at least **20 epochs**.

---

## Requirements
- **Python 3.10+**
- **PyTorch-based**, object-oriented programming, and clearly commented code
- **Pylint score:** 10/10
- **Writeup:** Completed in Overleaf using LaTeX, titled  
  _“Image Classification with Vision Transformers: An Experimental Study”_

---

## Submission
Include the following in your submission:
- All code in a folder named **`vision_transformer`**
- The experimental writeup titled  
  **`Image Classification with Vision Transformers: An Experimental Study`** saved as `experiments.pdf`
- **Training logs** clearly showing training/validation progress (included in notebook or separate log file)

**Minimum Performance Requirement:**  
Achieve at least **65% classification accuracy** on the test set.  
Penalties may apply for lower performance.
