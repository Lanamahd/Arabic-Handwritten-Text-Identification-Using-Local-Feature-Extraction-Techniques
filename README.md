# Arabic Handwritten Text Identification Using Local Feature Extraction Techniques

This repository focuses on the implementation of local feature extraction techniques for identifying Arabic handwritten text. By leveraging advanced image processing, feature extraction, and machine learning models, the project aims to classify handwritten Arabic text with improved accuracy and efficiency.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Installation and Usage](#installation-and-usage)
6. [File Descriptions](#file-descriptions)
7. [Conclusion](#conclusion)
8. [Technologies Used](#technologies-used)
9. [License](#license)

---

## Project Overview

Arabic handwritten text identification is a complex task due to the unique characteristics of Arabic script, including its cursive nature, varying shapes, and contextual dependencies. This project explores local feature extraction techniques, such as ORB (Oriented FAST and Rotated BRIEF), combined with machine learning models like Support Vector Machines (SVM) and K-Nearest Neighbors (KNN), to address these challenges.

The pipeline includes preprocessing, feature extraction, clustering, visualization, and classification of handwritten text images. The project also evaluates the performance of the models using metrics like accuracy, confusion matrices, and classification reports.

---

## Dataset

The dataset used for this project consists of images of Arabic handwritten text. These images are organized in the `isolated_words_per_user` directory and include:
- **Training Images**: Used for feature extraction and model training.
- **Test Images**: Used for evaluating model performance.

---

## Methodology

### 1. Data Preprocessing
- Normalization of input images.
- Resizing images to a uniform size.
- Data augmentation to improve model generalization.

### 2. Feature Extraction
- **ORB (Oriented FAST and Rotated BRIEF)**: Used for local feature extraction.
- **Clustering**: KMeans clustering to create visual words.
- **TF-IDF Weighting**: Applied to encode visual words into feature vectors.

### 3. Classification
- **Support Vector Machines (SVM)**: Trained using TF-IDF feature vectors for classification.
- **K-Nearest Neighbors (KNN)**: Used as an alternative classification model.

### 4. Dimensionality Reduction
- **Principal Component Analysis (PCA)**: Reduces feature dimensions for clustering and visualization.
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Visualizes data clusters in a lower-dimensional space.

### 5. Evaluation Metrics
- Accuracy score.
- Confusion matrix.
- Classification report.

---

## Results

The models achieved varying levels of accuracy with the dataset:
- **SVM Accuracy**: ~13% on the test set.
- **KNN Performance**: Evaluated as an alternative method.
- **Visualizations**: Clustering results visualized using t-SNE.

The relatively low accuracy indicates potential areas for improvement, such as feature engineering, hyperparameter tuning, and leveraging more robust datasets.

---

## Installation and Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.8 or later.
- Required libraries: `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`.

Install dependencies using:
    ```
    pip install numpy matplotlib scikit-learn opencv-python
    ```
    
### Running the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/Lanamahd/Arabic-Handwritten-Text-Identification-Using-Local-Feature-Extraction-Techniques.git
    cd Arabic-Handwritten-Text-Identification-Using-Local-Feature-Extraction-Techniques

2. Open and run the Jupyter Notebooks:

  - **Part 1**: `PART_ONE_Assignment2.ipynb` (Preprocessing and feature extraction).
  - **Part 2**: `AHAWPassignment-FINAL.ipynb` (Model training and evaluation).

3. Follow the instructions in each notebook to preprocess the data, extract features, train the models, and evaluate their performance.

---

## File Descriptions

### **`PART_ONE_Assignment2.ipynb`**
- Contains preprocessing steps and feature extraction using the ORB algorithm.

### **`AHAWPassignment-FINAL.ipynb`**
- Covers clustering, visualization, SVM training, and evaluation.

### **`CVassignment2 (2).pdf`**
- A detailed report summarizing the project objectives, methods, and results.

### **`isolated_words_per_user/`**
- Directory containing the dataset of Arabic handwritten text images.

---

## Conclusion

This project demonstrates the challenges and potential solutions in identifying Arabic handwritten text using local feature extraction techniques. Through a comprehensive pipeline involving preprocessing, feature extraction, clustering, and classification, the project highlights the application of robust models such as SVM and KNN. Despite achieving modest accuracy, the results underline the complexity of Arabic script recognition and the need for further optimization, including improved feature engineering, larger datasets, and hyperparameter tuning. Future work can explore deep learning-based approaches to enhance performance and accuracy. This repository serves as a valuable foundation for advancing Arabic handwritten text identification research.

---

## Technologies Used

- **Programming Language**: Python
- **Libraries**: OpenCV, scikit-learn, Matplotlib
- **Development Environment**: Jupyter Notebook

---

## License

This repository is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
