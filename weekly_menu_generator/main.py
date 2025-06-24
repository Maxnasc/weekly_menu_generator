# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Importando dados
df_main_menu = pd.read_csv('data/main_menu.csv')

# Retirando a coluna de id (útil apenas para o banco de dados)
df_main_menu = df_main_menu.drop(columns='id')

# Trocando os valores NaN por 0 (significa sem informação)
df_main_menu = df_main_menu.fillna(0)

# Convertendo o df para numpy, de forma a facilitar
main_meu_array = df_main_menu.to_numpy()

X = []
y = []

for i in range(len(main_meu_array)-13):
    entrada = main_meu_array[i:i+7]
    saida = main_meu_array[i+7:i+14]
    X.append(entrada)
    y.append(saida)
    
X = np.array(X)
y = np.array(y)

# Separando os dados em treino e teste
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)