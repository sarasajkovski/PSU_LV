import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('car_processed.csv')

# 1. Koliko mjerenja (automobila) je dostupno u datasetu?
print(f'Broj automobila u datasetu: {len(df)}')

# 2. Kakav je tip pojedinog stupca u dataframeu?
print("\nTipovi podataka po stupcima:")
print(df.dtypes)

# 3. Koji automobil ima najveću cijenu, a koji najmanju?
max_price = df.loc[df['selling_price'].idxmax()]
min_price = df.loc[df['selling_price'].idxmin()]
print("\nAutomobil s najvećom cijenom:")
print(max_price[['name', 'selling_price']])
print("\nAutomobil s najmanjom cijenom:")
print(min_price[['name', 'selling_price']])

# 4. Koliko automobila je proizvedeno 2012. godine?
count_2012 = df[df['year'] == 2012].shape[0]
print(f"\nBroj automobila iz 2012.: {count_2012}")

# 5. Koji automobil je prešao najviše kilometara?
most_km = df.loc[df['km_driven'].idxmax()]
print("\nAutomobil koji je prešao najviše kilometara:")
print(most_km[['name', 'km_driven']])

# 6. Koliko najčešće automobili imaju sjedala?
most_common_seats = df['seats'].mode()[0]
print(f"\nNajčešći broj sjedala: {most_common_seats}")

# 7. Prosječna kilometraža dizel vs benzinski automobili
mean_km_by_fuel = df.groupby('fuel')['km_driven'].mean()
print("\nProsječna prijeđena kilometraža po vrsti goriva:")
print(mean_km_by_fuel)

# BONUS: par grafova za bolji uvid
plt.figure(figsize=(8, 5))
sns.histplot(df['selling_price'], bins=30, kde=True)
plt.title('Distribucija cijena automobila (logaritmirana)')
plt.xlabel('Cijena')
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='year', order=sorted(df['year'].unique()))
plt.title('Broj automobila po godini proizvodnje')
plt.xticks(rotation=45)
plt.show()
