# Heart Disease Prediction

This project aims to predict the presence of heart disease in a patient using machine learning techniques. The model is trained on a dataset that contains various medical attributes.

## Dataset

The dataset contains the following columns:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalach`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`
- `target` (0 = No Heart Disease, 1 = Heart Disease)

## Project Structure

- `heart_disease_model.sav`: The trained logistic regression model saved using pickle.
- `app.py`: The Streamlit application for predicting heart disease.
- `README.md`: This file.
- Other required files and directories for the project.

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/heart-disease-prediction.git
    cd heart-disease-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Model Training

To train the model, follow these steps:

1. Load the dataset and perform basic data exploration:
    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    import pickle

    # Load data
    heart_data = pd.read_csv('heart.csv')

    # Data exploration
    print(heart_data.head())
    print(heart_data.tail())
    print(heart_data.shape)
    print(heart_data.info())
    print(heart_data.isnull().sum())
    print(heart_data.describe())
    print(heart_data['target'].value_counts())
    ```

2. Split the dataset into features and target:
    ```python
    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']

    print(X)
    print(Y)
    ```

3. Split the data into training and testing sets:
    ```python
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    print(X.shape, X_train.shape, X_test.shape)
    ```

4. Train the logistic regression model:
    ```python
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    ```

5. Evaluate the model:
    ```python
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy on Training data : ', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy on Test data : ', test_data_accuracy)
    ```

6. Save the trained model:
    ```python
    filename = 'heart_disease_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    ```

## Predicting Heart Disease

To predict heart disease for a new patient, use the following code:
```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Load the model
loaded_model = pickle.load(open('heart_disease_model.sav', 'rb'))
prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
