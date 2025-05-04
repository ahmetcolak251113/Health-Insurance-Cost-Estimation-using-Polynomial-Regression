import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Veriyi yükle
data_set = pd.read_csv('insurance.csv')
data_set = data_set.drop(columns=['children', 'region'])

# Giriş ve çıkışları ayır
X = data_set[["age", "sex", "bmi", "smoker"]]
y = data_set["charges"]

# Kategorik verileri sayıya çevir
X = data_set[["age", "sex", "bmi", "smoker"]].copy()
X["sex"] = X["sex"].map({"male": 1, "female": 0})
X["smoker"] = X["smoker"].map({"no": 0, "yes": 1})

poly_reg = PolynomialFeatures(degree=10)
X_poly = poly_reg.fit_transform(X)
Lin_reg = LinearRegression()
Lin_reg.fit(X_poly, y)
y_len = len(y.index)
y_pred = Lin_reg.predict(X_poly)
print(y_pred)

for i in range(y_len):
    print(f'| Real Value: {y[i]} | Predict: {y_pred[i]} | Error: {abs(y[i] - y_pred[i])} |')

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 15))
ax0.scatter(X["age"], y, color='blue', label='Gerçek Değer')
ax0.scatter(X["age"], y_pred, color='red', label='Tahmin')
ax0.set_title('Age - Charges Relationship')
ax0.set_xlabel("Age")
ax0.set_ylabel("Charges")
ax0.legend()
ax1.scatter(X["bmi"], y, color='blue', label='Gerçek Değer')
ax1.scatter(X["bmi"], y_pred, color='red', label='Tahmin')
ax1.set_title('BMI - Charges Relationship')
ax1.set_xlabel("BMI")
ax1.set_ylabel("Charges")
ax1.legend()
ax2.scatter(X["smoker"], y, color='blue', label='Gerçek Değer')
ax2.scatter(X["smoker"], y_pred, color='red', label='Tahmin')
ax2.set_title('Smoker - Charges Relationship')
ax2.set_xlabel("Smoker")
ax2.set_ylabel("Charges")
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Non-smoker', 'Smoker'])
ax2.legend()
ax3.scatter(X['sex'], y, color='blue', label='Gerçek Değer')
ax3.scatter(X['sex'], y_pred, color='red', label='Tahmin')
ax3.set_title('Sex - Charges Relationship')
ax3.set_xlabel("Sex")
ax3.set_ylabel("Charges")
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Female', 'Male'])
ax3.legend()
plt.tight_layout()
plt.show()

def calculator():
    age = int(input("Age: "))
    bmi = float(input("BMI: "))
    smoker = int(input("Smoker (yes: 1 / no: 0): "))
    sex = int(input("Sex (Male: 1 / Female: 0): "))
    return age, bmi, smoker, sex

age, bmi, smoker, sex = calculator()
input_df = pd.DataFrame([[age, sex, bmi, smoker]], columns=X.columns)
data_to_be_estimated_poly = poly_reg.transform(input_df)
predicted_value = Lin_reg.predict(data_to_be_estimated_poly)
print(f"Predicted value: {predicted_value[0]:.2f}$")

