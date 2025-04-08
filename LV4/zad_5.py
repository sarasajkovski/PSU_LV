import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import numpy as np

df = pd.read_csv('car_processed.csv')
df_clean = df.drop(columns=['name'])
print("Numerički stupci:", df_clean.select_dtypes(include=np.number).columns.tolist())

X = df_clean.drop(columns=['selling_price'])  # Ulazne značajke
y = df_clean['selling_price']                 # Ciljna varijabla

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#4
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("Parametri modela (koeficijenti):")
print(model.coef_)
print("Presretanje (intercept):", model.intercept_)

#5
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\nEvaluacija na TRAIN skupu:")
print("MAE:", mean_absolute_error(y_train, y_train_pred))
print("MSE:", mean_squared_error(y_train, y_train_pred))
print("R2 score:", r2_score(y_train, y_train_pred))
print("Max error:", max_error(y_train, y_train_pred))
print("\nEvaluacija na TEST skupu:")
print("MAE:", mean_absolute_error(y_test, y_test_pred))
print("MSE:", mean_squared_error(y_test, y_test_pred))
print("R2 score:", r2_score(y_test, y_test_pred))
print("Max error:", max_error(y_test, y_test_pred))

#6
#Kada smanjujemo broj ulaznih veličina, model koristi manje informacija, što može dovesti do veće
#pogreške jer gubi korisne značajke. Ako uključimo previše značajki, uključujemo i šum, što može 
#dovesti do lošije generalizacije na testnom skupu. 

