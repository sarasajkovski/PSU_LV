import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\student\\Desktop\\lV5\\occupancy_processed.csv')
feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'
class_names = ['Slobodna', 'Zauzeta']

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

plt.figure()
for class_value in np.unique(y):
    mask = y == class_value
    plt.scatter(X[mask, 0], X[mask, 1], label=class_names[class_value])
print(df)
plt.xlabel('S3_Temp')
plt.ylabel('S5_CO2')
plt.title('Zauzetost prostorije')
plt.legend()
plt.show()

#a) Pokrenite skriptu i pogledajte dobiveni dijagram raspršenja. Što primjećujete?
#Primjećujem da postoji razlika u raspodjeli klasa; vizualno su klase djelomično odvojeno 
#što znači da klasifikator može imati dobru točnost.

#b) Koliko podatkovnih primjera sadrži učitani skup podataka?- 10, 129

#c) Kakva je razdioba podatkovnih primjera po klasama?
#Klasa "Slobodna" (0): 8228 primjera – 81.23%
#Klasa "Zauzeta" (1): 1901 primjera – 18.77%
#- Podaci su neuravnoteženi 
