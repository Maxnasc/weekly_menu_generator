import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import random

def get_menu_data():
    # Importando dados
    df_main_menu = pd.read_csv('data/main_menu.csv')
    # Trocando os valores NaN por 0 (significa sem informação)
    df_main_menu = df_main_menu.fillna(0)
    return df_main_menu

def predict(df_menu, start_day):
    def get_random_week_menu(np_menu):
        def get_random_day(day):
            # Filtrando dados
            ultima_coluna = np_menu[:, -1]
            condicao = (ultima_coluna == day)
            np_menu_filtered = np_menu[condicao]
            # Amostrando aleatóriamente
            return np_menu_filtered[random.randint(0,np_menu_filtered.shape[0]), :]
        
        # Obter uma semana formada por cardápios de dias escolhidos aleatóriamente
        random_week_list = [[]]
        day = start_day
        for i in range(7):
            random_week_list[0].append(get_random_day(day))
            if day == 7:
                day = 1
            else:
                day+=1
        random_week = np.array(random_week_list)
        return random_week
    
    # Importando o modelo keras
    model = load_model('weekly_menu_generator/model/modelo_cardapio.keras')
    
    # Convertendo o df para numpy, de forma a facilitar
    df_menu_wo_id = df_menu.drop(columns='id')
    main_menu_array = df_menu_wo_id.to_numpy()
    
    # Últimos 7 dias
    last_7_days = np.array([main_menu_array[len(main_menu_array)-7:len(main_menu_array)]])
    # Semana aleatoria
    random_week = get_random_week_menu(main_menu_array)
    
    
    # Fazendo a previsão
    previsao = model.predict(random_week)

    previsao_discretizada = np.rint(previsao).astype(int)

    # Substituindo os valores dos dias por uma sequencia de dias
    ultimo_dia = 7 # Sábado
    # Construir os próximos 7 dias
    dias_sequenciais = [(ultimo_dia + i - 1) % 7 + 1 for i in range(1,8)]
    # Substituição dos dias sequanciais

    previsao_discretizada[:,:,5] = dias_sequenciais
    
    # Convertendo o array em 2D e retornando a previsão para a planilha de menu
    colunas = ['basis', 'follow_up', 'protein', 'garrison', 'optional', 'day']
    previsao_discretizada_2d = previsao_discretizada.reshape(-1,6)
    
    df_previsao = pd.DataFrame(previsao_discretizada_2d, columns=colunas)
    
    # Obter o último índice do df original
    proximo_id = df_menu['id'].max() + 1
    # Calcular índices do cardápio predito
    novos_ids = np.arange(proximo_id, proximo_id+len(df_previsao))
    # Inserir índices dentro do df_previsao
    df_previsao.insert(0, 'id', novos_ids)
    
    df_menu_appended = pd.concat([df_menu, df_previsao], ignore_index=True)
    df_menu_appended = df_menu_appended.replace(0, np.nan)
    # df_menu_appended.to_csv('data/main_menu.csv') # Ignorado por enquanto
    
    return previsao_discretizada

def convert(previsao):
    # Importar todas as planilhas de conversão
    df_basis_template = pd.read_csv('data/Basis.csv')
    df_days_template = pd.read_csv('data/Days.csv')
    df_follow_up_template = pd.read_csv('data/Follow_up.csv')
    df_garrison_template = pd.read_csv('data/Garrison.csv')
    df_optional_template = pd.read_csv('data/Optional.csv')
    df_protein_template = pd.read_csv('data/Protein.csv')
    
    colunas = ['basis', 'follow_up', 'protein', 'garrison', 'optional', 'day']
    previsao_2d = previsao.reshape(-1,6)
    
    # Criando o dataframe com a previsão
    df = pd.DataFrame(previsao_2d, columns=colunas)
    
    # Criando as séries de dados para converter os números em nomes de cada componentes
    map_basis = df_basis_template.set_index('id_basis')['basis_name']
    map_days = df_days_template.set_index('id_day')['day_name']
    map_follow_up = df_follow_up_template.set_index('id_follow_up')['follow_up_name']
    map_garrison = df_garrison_template.set_index('id_garrison')['garrison_name']
    map_optional = df_optional_template.set_index('id_optional')['optional']
    map_protein = df_protein_template.set_index('id_protein')['protein_name']
    
    # Utilizando as séries de dados para converter os códigos em números
    df['basis'] = df['basis'].map(map_basis)
    df['follow_up'] = df['follow_up'].map(map_follow_up)
    df['protein'] = df['protein'].map(map_protein)
    df['garrison'] = df['garrison'].map(map_garrison)
    df['optional'] = df['optional'].map(map_optional)
    df['day'] = df['day'].map(map_days)
    
    return df

def save_prediction(df):
    df.to_excel('weekly_menu_generator/predictions/cardapio_predito.xlsx')

def main():
    df_menu = get_menu_data()
    result = predict(df_menu, 6)
    result_converted = convert(result)
    save_prediction(result_converted)
    print(result_converted)
    
if __name__=='__main__':
    main()