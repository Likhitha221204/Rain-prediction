# Rain-prediction
This model predicts next-day rainfall using a weather dataset. It includes data cleaning, handling missing values, encoding categorical variables, and feature scaling. A logistic regression model is trained and evaluated using metrics like ROC-AUC and confusion matrix for performance assessment.


# Rainfall Prediction Model

## Overview
This project aims to predict whether it will rain tomorrow using a comprehensive weather dataset. The model performs data cleaning, feature engineering, and trains a logistic regression model to make predictions.

## Dataset
The dataset used in this project is a weather dataset containing various features such as temperature, humidity, wind speed, and more. It can be accessed [here](https://drive.google.com/drive/folders/1_uCLNkKkoI2UtHPZocdoovG5TnYFrSyV?usp=sharing).

## Key Features
- **Data Cleaning**: Handling missing values and duplicates.
- **Feature Engineering**: Encoding categorical variables and scaling features.
- **Model Training**: Using logistic regression to predict rainfall.
- **Performance Evaluation**: Metrics include ROC-AUC and confusion matrix.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/rainfall-prediction.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Load the dataset:
    ```python
    data = pd.read_csv('path/to/weatherAUS.csv')
    ```
2. Preprocess the data and train the model:
    ```python
    # Data preprocessing steps
    data.drop_duplicates(inplace=True)
    data.fillna(data.mode().iloc[0], inplace=True)

    # Encode categorical features and scale data
    # ...

    # Train logistic regression model
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    ```

3. Evaluate the model:
    ```python
    from sklearn.metrics import roc_auc_score, confusion_matrix
    y_pred = clf.predict(X_test)
    print('ROC-AUC Score:', roc_auc_score(y_test, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License
This project is licensed under the MIT License.

---

Feel free to customize it further based on your specific project details and preferences!
