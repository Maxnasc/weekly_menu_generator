import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

def predict():
    # Importando o modelo keras
    model = load_model('weekly_menu_generator/model/modelo_cardapio.keras')
    last_7_days = np.load('weekly_menu_generator/model/entrada_final.npy')
    
    previsao = model.predict(last_7_days)

    previsao_discretizada = np.rint(previsao).astype(int)

    # Substituindo os valores dos dias por uma sequencia de dias
    ultimo_dia = 7 # Sábado
    # Construir os próximos 7 dias
    dias_sequenciais = [(ultimo_dia + i - 1) % 7 + 1 for i in range(1,8)]
    # Substituição dos dias sequanciais

    previsao_discretizada[:,:,5] = dias_sequenciais
    
    return previsao_discretizada

def convert(previsao):
    # Importar todas as planilhas de conversão
    df_basis_template = pd.read_csv('data/Basis.csv')
    df_days_template = pd.read_csv('data/Days.csv')
    df_follow_up_template = pd.read_csv('data/Follow_up.csv')
    df_garrison_template = pd.read_csv('data/Garrison.csv')
    df_optional_template = pd.read_csv('data/Optional.csv')
    df_protein_template = pd.read_csv('data/Protein.csv')
    
    previsao_convertida = pd.DataFrame()
    

def main():
    # Importando o modelo keras
    model = load_model('weekly_menu_generator/model/modelo_cardapio.keras')
    last_7_days = np.load('weekly_menu_generator/model/entrada_final.npy')
    
    
