# Rozpoznawanie języka migowego
### Autor: Jakub Marciniak

## Opis

Projekt ma na celu impementację uczenia maszynowego mającego na celu wykrywanie liter w języku migowym. 


Do uczenia modelu wykorzystano stworzony wspólnie z grupą roku dataset zawierający wszystkie litery alfabetu migowego w ilościach proporcjonalnych.
W projekcie wykorzystano jedynie dane landmark we współrzędnych ramki, a nie world.
Ponadto wykorzystano dane z kolumny handeness.label, które zostały zenkodowane przy pomocy OneHotEncoder.
Na koniec wszyskie dane zostały ustandaryzowane przy pomocy StandardScaler.
Na danym zbiorze danych wyuczono model LogisticRegression. Parametry modelu dobrano przy pomocy GridSearch. Dane do uczenia zostały podzielone w proporcjach 95% zbiór uczący oraz 5% zbiór testowy oraz zastosowano flagę 'stratify'. 