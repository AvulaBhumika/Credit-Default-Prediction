# **Credit Default Prediction ML Pipeline**

![image](https://github.com/user-attachments/assets/01c4beeb-ef0a-4b3b-9ddc-53af9c200bfb)


## **Overview**
This project aims to predict loan defaulters using machine learning techniques, leveraging **XGBoost** as the primary model. The pipeline includes data extraction, exploratory data analysis (EDA), feature importance analysis, model training, evaluation, and storing predictions in a database.

## **Project Workflow**
1. **Loading Data from Database**
   - Connect to the database and extract loan application data.
   - Perform initial data checks to ensure integrity and consistency.
   
2. **Exploratory Data Analysis (EDA)**
   - Handle missing values and data inconsistencies.
   - Visualize key variables such as loan amount, interest rates, income levels, and credit history.
   - Analyze class imbalance in the target variable (default vs. non-default).

3. **Feature Engineering & Insights**
   - Identify the most influential features using statistical analysis and domain knowledge.
   - One-hot encode categorical variables.
   - Normalize numerical features for better model performance.
   - Generate feature importance plots to identify key predictors of loan default.

4. **Model Building**
   - Implement **RandomForestClf** as the primary model.
   - Perform hyperparameter tuning using Randomizedsearch with XGB Classifier.
   - Train the model on historical loan application data.

5. **Model Evaluation**
   - Generate key classification metrics:
     - **Precision, Recall, F1-Score** (to measure accuracy on default vs. non-default)
     - **ROC-AUC Score** (to evaluate the model’s ability to distinguish between classes)
   - Visualize the **confusion matrix** to assess misclassification patterns.
   
6. **Saving Predictions to Database**
   - Store predicted default probabilities in a new table **‘Defaulters_Predicted’**.
   - Maintain loan IDs along with their respective predictions for tracking.

## **Installation & Setup**
### **Requirements**
Ensure you have the following installed:
```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn sqlalchemy 
```

### **Database Connection**
Modify the database connection string as needed:
```python
import sqlalchemy
engine = sqlalchemy.create_engine('postgresql://user:password@localhost:5432/db_name')
```


## **Results & Insights**
- Loan grade (D, E, F) is the most critical factor influencing defaults.
- Renters are more likely to default than homeowners.
- Higher debt-to-income ratios correlate with increased default risks.
- Borrowers with previous defaults have a significantly higher chance of defaulting again.

  ![image](https://github.com/user-attachments/assets/b0538352-e418-4ff9-a79e-595c3fadc827)


## **Next Steps**
✅ Deploy the model as an API for real-time predictions.
✅ Implement monitoring and alerting mechanisms for high-risk loans.
✅ Extend the analysis with deep learning models for improved accuracy.


