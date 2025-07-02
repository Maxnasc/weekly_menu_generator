# Importando bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Input, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, root_mean_squared_error
import seaborn as sns
import json

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
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Separando em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

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
    
    # Avaliando o modelo
    evaluate_model(model, X_test, y_test)
    
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

def evaluate_model(model, X_test, y_test):
    # Previsão do modelo
    y_pred = model.predict(X_test)
    y_pred = np.rint(y_pred).astype(int)
    metrics_result = {}
    
    # Cálculo geral de MAE
    mae = mean_absolute_error(y_test.reshape(-1, 6), y_pred.reshape(-1, 6))
    metrics_result['mae'] = mae
    print(f"MAE total: {mae:.2f}")
    
    # Listas de nomes dos campos
    campos = ['basis', 'follow_up', 'protein', 'garrison', 'optional', 'day']
    
    # Avaliação por campo    
    metrics_result['acc'] = {}
    metrics_result['rsme'] = {}
    for i, nome in enumerate(campos):
        acc = accuracy_score(y_test[:, :, i].flatten(), y_pred[:, :, i].flatten())
        rmse = root_mean_squared_error(y_test[:, :, i].flatten(), y_pred[:, :, i].flatten())
        print(f"Acurácia em {nome}: {acc:.2%}")
        print(f"RMSE para {nome}: {rmse:.2f}")
        metrics_result['acc'][nome] = f'{acc:.2%}'
        metrics_result['rsme'][nome] = rmse
        
        # Plotar matriz de confusão
        plot_confusion(y_test, y_pred, i, title=f'Matriz de Confusão - {nome}')
    with open('weekly_menu_generator/model/metrics/metrics.json', 'w') as arq:
        json.dump(metrics_result, arq)

# Função auxiliar fora do escopo local
def plot_confusion(y_true, y_pred, field_index, title=''):
    cm = confusion_matrix(y_true[:, :, field_index].flatten(),
                          y_pred[:, :, field_index].flatten())
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title or f'Confusão campo {field_index}')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(f'weekly_menu_generator/model/metrics/grafico_{title}')

def main():
    X, y, main_menu_array = data_import()
    history, model = train_model(X, y)
    plot_history(history)
    previsao = predict(main_menu_array, model)
    
    print(previsao)
    plt.show()
    
if __name__=='__main__':
    main()