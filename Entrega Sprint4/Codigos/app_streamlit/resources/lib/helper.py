import numpy as np
# from numpy.core.multiarray import where
import pandas as pd 
# import streamlit as st
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

def tratarPlanilha(file):
    ''' 
    Retorna a planilha de dados tratada.

    Esta função realiza a ingestão da planilha como DataFrame Pandas
    removendo linhas e colunas em branco, assim como linhas com valores faltantes
    e features irrelevantes para as análises em questão, adicionando novas features relevantes.

    Parâmetros
    ----------
    file : None ou streamlit.UploadedFile (subclasse de BytesIO)
        Arquivo resultante do widget streamlit.file_uploader

    Retorna
    -------
    DataFrame
        DataFrame Pandas com os dados após tratamento inicial.
    '''
    spreadsheet = pd.read_excel(file)

    spreadsheet.rename(columns={'Camada Superior': 'Camada_Superior'}, inplace=True)
    spreadsheet.rename(columns={'Camada Inferior': 'Camada_Inferior'}, inplace=True)

    # filtra 0 < valores < 5 da camada superior
    maiorQ0 = spreadsheet['Camada_Superior'] > 0
    spreadsheet = spreadsheet[maiorQ0]
    menorQ5 = spreadsheet['Camada_Superior'] < 5
    spreadsheet = spreadsheet[menorQ5]

    # # Remover colunas em branco
    spreadsheet.dropna(how='all', axis=1, inplace=True)
    # # Remover linhas em branco
    spreadsheet.dropna(how='all', axis=0, inplace=True)

    spreadsheet.dropna(how='any', axis=0, inplace=True)
    # # Reset Index
    spreadsheet.reset_index(drop=True, inplace=True)
    # # Usa a primeira coluna como header
    # spreadsheet.columns = spreadsheet.iloc[0]
    # # Drop linha com os nomes das colunas
    spreadsheet.drop(0, inplace=True)
    # # Reset Index
    spreadsheet.reset_index(drop=True, inplace=True)

    # # Drop colunas irrelevantes para análise
    spreadsheet.drop(columns=['Data Amostra', 'Lote mae', 'Lote producao', 'Unnamed: 0'], inplace=True)

    # SegProd = spreadsheet['Segmento_Prod'] == 'CC'
    # filtered_CC = spreadsheet[SegProd]
    # filtered_CC
    #
    # SegProd = spreadsheet['Segmento_Prod'] == 'LB'
    # filtered_LB = spreadsheet[SegProd]
    # filtered_LB

    # Criação do encoder
    onehotencoder = OneHotEncoder()

    # remodelar a matriz de país 1-D para 2-D, já que fit_transform espera 2-D e finalmente ajusta o objeto
    encoded_array = onehotencoder.fit_transform(spreadsheet['Segmento_Prod'].values.reshape(-1, 1)).toarray()

    # Criando novas colunas com os nomes corretos para adicionar ao dataframe original
    col_tipo_material = pd.DataFrame(encoded_array, columns=onehotencoder.get_feature_names_out(['Segmento_Prod']))

    # Adicionando as colunas ao dataframe original
    spreadsheet = pd.concat([spreadsheet, col_tipo_material], axis=1)

    # Drop coluna original de Tipo de Material
    spreadsheet = spreadsheet.drop(['Segmento_Prod'], axis=1)




    return spreadsheet

def gerarEquacao(inputs, out, df, descarte=False):
    '''
    Realiza a regressão linear utilizando métodos da biblioteca statsmodels.

    Realiza a regressão linear utilizando as técnias OLS, GLS e GLSAR da 
    biblioteca statsmodels aplicada ao dataset fornecido, utilizando como variável
    dependente o parâmetro ``out`` e como entradas o parâmetro ``inputs``. 

    Caso o parâmetro ``descarte`` seja ``True``, realiza um segundo fit dos dados removendo
    das entradas os parâmetros cujo valor-p foi maior que 0,05 no primeiro fit.

    Parâmetros
    ----------
    inputs : lista de str 
        Lista contendo o nome das colunas do dataframe a serem utilizadas como entradas.
    out : str
        Nome da coluna a ser utilizada como variável dependente.
    df : pandas.DataFrame
        Pandas DataFrame contendo os dados a serem utilizados.
    descarte : bool
        Flag utilizada para indicar se deve ser feito o descarte das variáveis de menor impacto na regressão.

    Retorna
    -------
    FITTED_MODELS : dict of {'OLS' : statsmodels.api.OLS, 'GLS' : statsmodels.api.GLS, 'GLSAR' : statsmodels.api.GLSAR} 
        Dicionário contendo os nomes dos modelos como chaves e os modelos após o fit aos dados como itens.
    '''
    # Criar DataFrames/Series de entradas e saídas
    y = df[out]
    X = df[inputs]

    # Adiciona uma constante inicial ao projeto
    X = sm.add_constant(X) 

    # Cria dicionário com os modelos a serem utilizados
    MODELS = {'OLS': sm.OLS, 'GLS': sm.GLS, 'GLSAR': sm.GLSAR}
    # Dicionário vazio para armazenar os modelos após o fit
    FITTED_MODELS = {}

    # Fit inicial dos modelos
    for model in MODELS:
        FITTED_MODELS[model] = MODELS[model](y, X).fit()
    # Verifica se o descarte de features deve ser feito
    if descarte:
        for model in MODELS:
            # Verifica o valor-p, caso maior que 0,05 o index da feature é armazenado na lista
            mdl_discard_index = [ X.columns[ind] for ind, pvalue in enumerate(FITTED_MODELS[model].pvalues) if pvalue > 0.05 ]
            # Realiza novo fit do modelo, descartando as features cujos index estão armazenados em mdl_discard_index
            FITTED_MODELS[model] = MODELS[model](y, X.drop(mdl_discard_index, axis=1)).fit()

    return FITTED_MODELS
