# 🚢 Titanic Survival Prediction

## Project Overview
This project predicts Titanic passenger survival using machine learning. It involves data preprocessing, feature engineering, and training a **Logistic Regression** model. Key steps include handling missing data, encoding categorical variables, and model evaluation. The goal is to build an accurate classifier that determines survival likelihood based on passenger characteristics like age, gender, and class. 🚢


---

## **📌 Task Objectives**
- Predict passenger survival using machine learning.
- Preprocess and clean the dataset for better accuracy.
- Train and evaluate a **Logistic Regression** model (or other models like Random Forest).
- Optimize the model for better predictions.

---

## **🛠 Steps to Run the Project**
### **1️⃣ Import Libraries**
```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report ,accuracy_score,mean_absolute_error,mean_squared_error,r2_score,confusion_matrix
import joblib

import warnings
warnings.filterwarnings('ignore')
```

### **2️⃣ Load & Explore Data**
```python
df=pd.read_csv("D:\\Programming\\Datasets\\Titanic Surival\\tested.csv")
print(df.head())  # Check the first few rows
print(df.info())  # Understand missing values
```

### **3️⃣ Data Cleaning & Preprocessing**
```python
# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin', 'Ticket', 'Name'], inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
```

### **4️⃣ Feature Selection & Splitting Data**
```python
X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
```

### **5️⃣ Train the Logistic Regression Model**
```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

### **6️⃣ Make Predictions & Evaluate**
```python
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### **7️⃣ Saving the Model with joblib**
```python
model_filename = 'model.joblib'

joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")
```


---

## **📊 Evaluation Criteria**

### **1️⃣ Functionality**  
✔ **Correctness:** The model successfully predicts survival with a reasonable accuracy.  
✔ **Data Handling:** Properly deals with missing values, feature encoding, and scaling.  
✔ **Model Performance:** Achieves a solid accuracy with meaningful evaluation metrics (Precision, Recall, F1-score).  
✔ **Reproducibility:** The code runs without errors and produces consistent results.  

---

### **2️⃣ Code Quality**  
✔ **Structure:** The code is modular, using functions for preprocessing, training, and evaluation.  
✔ **Readability:** Proper indentation, meaningful variable names, and clear comments.  
✔ **Efficiency:** Uses optimized Pandas and NumPy operations (avoids unnecessary loops).  
 
---

### **3️⃣ Innovation & Creativity**  
✔ **Feature Engineering:** Uses innovative feature selection (e.g., creating new features from existing ones).  
✔ **Model Optimization:** Tries different models (e.g., Logistic Regression, Random Forest, XGBoost) and hyperparameter tuning.  
✔ **Exploratory Data Analysis (EDA):** Provides meaningful visualizations (e.g., survival rate based on gender, class, and age).  

---

###  Key Sections:
**Data Loading**: The load_data() function loads the Titanic dataset from a specified path.  
**Preprocessing**: The preprocess_data() function handles missing data, encodes categorical variables, and prepares the dataset for modeling.  
**Data Splitting**: The split_data() function splits the dataset into training and testing sets (80% train, 20% test).  
**Model Training**: The train_model() function trains a logistic regression model using Scikit-learn.  
**Model Evaluation**: The evaluate_model() function provides accuracy, confusion matrix, and classification report metrics for model performance.  
**Main Execution**: The main() function ties everything together, ensuring a smooth flow from data loading to evaluation.








---

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  

## Setup Instructions

### 1️⃣ Install Dependencies

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
pip install -r requirements.txt
```

### 2️⃣ Prepare the Data

The dataset used is the [Titanic dataset from Kaggle](https://www.kaggle.com/c/titanic/data). Download the dataset and place it in the `data/` folder of the project.

```plaintext
data/
    train.csv
    test.csv
```

### 3️⃣ Preprocess Data

Before training the model, the data needs to be preprocessed. This includes handling missing values, encoding categorical variables, and scaling numeric features. The preprocessing steps can be found in the script `preprocess.py`.

### 4️⃣ Model Training

You can choose to train the model using various machine learning algorithms. Some popular choices include:

- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- XGBoost (for advanced performance)

Run the training script:

```bash
python train.py
```

### 5️⃣ Model Evaluation

After training the model, it is evaluated using various performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

The evaluation is conducted on the test set (separate from the training data), and results are printed in the console.


---

### **6️⃣ Make Predictions & Evaluate**

Now that the model is trained and tuned, we can use it to make predictions on new data and evaluate the results.

1. **Load the trained model**: You’ll load the saved model file, typically a `.pkl` (pickle) file.
   
   Example:

   ```python
   import pickle
   model = pickle.load(open('model.pkl', 'rb'))
   ```

2. **Make Predictions**: Use the trained model to predict survival outcomes on the test dataset or new passenger data.
   
   Example:

   ```python
   predictions = model.predict(X_test)
   ```

3. **Evaluate Predictions**: Compare the predictions to the actual values and assess model performance.
   
   Example:

   ```python
   from sklearn.metrics import accuracy_score, classification_report

   print("Accuracy:", accuracy_score(y_test, predictions))
   print("Classification Report:\n", classification_report(y_test, predictions))
   ```

4. **Save Predictions**: Optionally, you can save the predictions to a CSV file for submission to Kaggle or to review the results.
   
   Example:

   ```python
   pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions}).to_csv('predictions.csv', index=False)
   ```

---


## **Results**  

### **Model Performance**  

| Metric            | Score  |
|------------------|--------|
| **Accuracy**      | 100%   |
| **Precision**     | 1.00   |
| **Recall**        | 1.00   |
| **F1-Score**      | 1.00   |
| **ROC-AUC Score** | 1.00   |

### **Confusion Matrix**  

| Actual \ Predicted | Survived (1) | Not Survived (0) |
|-------------------|-------------|----------------|
| **Survived (1)**   | 100% (TP)    | 0% (FN)        |
| **Not Survived (0)** | 0% (FP)     | 100% (TN)      |

### **Conclusion**  

The model achieved **100% accuracy**, meaning it perfectly classified every passenger’s survival status. However, such a high score suggests potential **overfitting** or **data leakage**. Further validation, such as cross-validation on unseen data, is essential to ensure the model generalizes well to real-world scenarios.  

