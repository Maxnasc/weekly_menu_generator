# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, Dropout
import matplotlib.pyplot as plt


def data_import():
    # Importando dados
    df_main_menu = pd.read_csv('data/main_menu.csv')

    # Retirando a coluna de id (útil apenas para o banco de dados)
    df_main_menu = df_main_menu.drop(columns='id')

    # Trocando os valores NaN por 0 (significa sem informação)
    df_main_menu = df_main_menu.fillna(0)

    # Convertendo o df para numpy, de forma a facilitar
    main_menu_array = df_main_menu.to_numpy()

    X = []
    y = []

    for i in range(len(main_menu_array)-13):
        entrada = main_menu_array[i:i+7]
        saida = main_menu_array[i+7:i+14]
        X.append(entrada)
        y.append(saida)
        
    X = np.array(X)
    y = np.array(y)
    
    return X, y, main_menu_array

def train_model(X, y):
    # Separando os dados em treino e teste
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    
    # Dimensões de entrada
    n_dias = X_train.shape[1]
    n_features = X_train.shape[2]

    # Configuração do modelo
    model = Sequential([
        Input(shape=(n_dias, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        TimeDistributed(Dense(64, activation='relu')),
        Dropout(0.2),
        TimeDistributed(Dense(n_features, activation='relu')),
    ])

    # Configuração do otimizador do modelo
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    # Treinamento do modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=8,
    )
    
    model.save('weekly_menu_generator/model/modelo_cardapio.keras')
    
    return history, model

def predict(main_menu_array, model):
    # Fazendo previsão dos próximos 7 dias
    entrada_final = main_menu_array[-7:].reshape(1,7,6)
    np.save('weekly_menu_generator/model/entrada_final.npy', entrada_final)
    previsao = model.predict(entrada_final)

    previsao_discretizada = np.rint(previsao).astype(int)

    # Substituindo os valores dos dias por uma sequencia de dias
    ultimo_dia = int(main_menu_array[-1][-1])
    # Construir os próximos 7 dias
    dias_sequenciais = [(ultimo_dia + i - 1) % 7 + 1 for i in range(1,8)]
    # Substituição dos dias sequanciais

    previsao_discretizada[:,:,5] = dias_sequenciais
    

    return previsao_discretizada

def plot_history(history):
    plt.plot(history.history["loss"], label="Treinamento")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title("Erro durante o treinamento")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()
    plt.savefig('weekly_menu_generator/model/history.png')
    plt.show()

def main():
    X, y, main_menu_array = data_import()
    history, model = train_model(X, y)
    plot_history(history)
    previsao = predict(main_menu_array, model)
    
    print(previsao)
    
if __name__=='__main__':
    main()