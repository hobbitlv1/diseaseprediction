# Heart Disease Prediction Model

## Overview

This repository contains a comprehensive heart disease prediction model developed using a combination of machine learning algorithms. The model leverages Logistic Regression, Random Forest, and Support Vector Machine classifiers, combined through a voting mechanism to enhance prediction accuracy and generalization. This approach capitalizes on the strengths of each algorithm while mitigating their individual weaknesses, resulting in a robust and reliable predictive model.

## Key Features

### Data Loading and Preprocessing

The dataset used for this model includes various vital signs of patients, which are utilized as features to predict the presence of heart disease. The data undergoes preprocessing, including feature selection to focus on the most relevant attributes, ensuring the highest prediction scores during the model's learning process.

### Model Creation

The model employs ensemble learning techniques, incorporating:
- **Logistic Regression** for simplicity and interpretability.
- **Random Forest** to capture non-linear relationships within the data.
- **Support Vector Machine (SVM)** to handle complex decision boundaries.

Balanced class weights are used to address potential class imbalance in the dataset, enhancing the model's performance.

### Hyperparameter Tuning

The model undergoes hyperparameter tuning using GridSearchCV, optimizing parameters such as regularization strength (C) for Logistic Regression, the number of estimators and maximum depth for Random Forest, and regularization parameters (C and gamma) for SVM. This ensures the model is fine-tuned for optimal performance.

### Model Evaluation

The model's performance is evaluated using a variety of metrics, including precision, recall, F1-score, and AUC-ROC. Cross-validation scores provide a comprehensive view of the model's generalization capabilities, ensuring its reliability across different datasets.

### Visualization of Results

Various visualizations are created to interpret the model's performance, including:
- **Confusion Matrix**: To understand the true positive, true negative, false positive, and false negative predictions.
- **ROC Curve**: To assess the model's ability to distinguish between classes.

### Feature Importance Analysis

Feature importance analysis is conducted to identify which factors contribute most significantly to the model's predictions. This helps in understanding the underlying patterns and relationships within the data.

## Repository Structure

- **Notebook 1.ipynb**: Jupyter notebook containing the complete code for data loading, preprocessing, model creation, evaluation, and visualization.
- **README.md**: Provides an overview of the project, instructions for usage, and descriptions of the files and directories.

## Getting Started

1. **Clone the Repository**: 
   ```bash
   git clone https://github.com/hobbitlv1/diseaseprediction.git
   ```

2. **Install Dependencies**:
   Ensure you have the required libraries installed. You can install them using pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. **Run the Notebook**:
   Open and run the `Notebook 1.ipynb` in Jupyter Notebook or any compatible environment.

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

For more detailed information, please refer to the Jupyter notebook and the code comments within the repository.
