Health Insurance Cost Estimation using Polynomial Regression

This project demonstrates how polynomial regression can be applied to a real-world dataset to predict health insurance charges based on personal and lifestyle information.

Files:
- insurance.csv: Dataset containing demographic and health-related features.
- healthinsurancepolyreg.py: Python script that:
  - Preprocesses the data
  - Converts categorical variables
  - Trains a polynomial regression model
  - Visualizes results
  - Provides a command-line interface for custom predictions

Dataset Description (insurance.csv):
Each row in the dataset represents a person's health profile and the corresponding insurance charge.

Columns:
- age      : Age of the individual
- sex      : Gender ('male' or 'female')
- bmi      : Body Mass Index
- children : Number of children (removed in preprocessing)
- smoker   : Smoking status ('yes' or 'no')
- region   : Region in the US (removed in preprocessing)
- charges  : Annual insurance cost in USD

Features Used (after preprocessing):
- age      (numeric)
- sex      (binary: male = 1, female = 0)
- bmi      (numeric)
- smoker   (binary: yes = 1, no = 0)

How It Works:
1. Polynomial Regression is applied (degree = 10) to capture non-linear relationships.
2. The model predicts the 'charges' column (health insurance cost).
3. Multiple scatter plots are generated to compare real vs predicted values for:
   - Age
   - BMI
   - Smoking status
   - Gender
4. A user can enter their own data to receive a cost prediction via the command line.

How to Run:
1. Make sure you have Python installed.
2. Install required libraries:
   pip install pandas matplotlib scikit-learn
3. Run the script:
   python healthinsurancepolyreg.py
4. Follow the input prompts to get a predicted insurance cost.

Output Example:
- Visualization of model predictions vs actual charges
- Printed error metrics for each instance
- Interactive prediction via terminal

Notes:
- The model uses a very high polynomial degree (10), which may lead to overfitting.
- This script is for educational purposes and can be optimized further using cross-validation and model tuning.
