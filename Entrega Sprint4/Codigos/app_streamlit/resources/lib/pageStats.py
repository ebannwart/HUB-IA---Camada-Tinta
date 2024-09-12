import re
# import os
# import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error#, mean_absolute_percentage_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, MultiTaskElasticNetCV, MultiTaskLassoCV, RANSACRegressor, Ridge, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import resources.lib.helper as helper

def gerarEstatisticas(modelo, name, X, y, train_size):
    '''
    Retorna DataFrame contendo estatísticas do modelo informado nos parâmetros.

    Esta função realiza o treinamento do ``modelo``, com as entradas e saídas informadas
    (``X`` e ``y``, respectivamente) utilizando ``name`` como index e a 
    porcentagem de Hold Out ``train_size``. Após o treinamento são calculadas estatísticas,
    que são retornadas como um DataFrame Pandas.

    Parâmetros
    ----------
        modelo : modelo sklearn
            Modelo sklearn que implementa as métricas utilizadas.
        name : str
            Nome a ser utilizado para o modelo.
        X : pandas.DataFrame
            DataFrame contendo os dados de entrada. 
        y : pandas.DataFrame
            DataFrame contendo os dados de saída. 
        train_size : float
            Porcentagem de dados reservados para treinamento do modelo.

    Retorna
    -------
        results : pandas.DataFrame
            DataFrame contendo as estatísticas e utilizando ``name`` como index.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r_squared = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    # mape = mean_absolute_percentage_error(y_test, y_pred)
    acc = modelo.score(X_test, y_test)

    results = pd.DataFrame(
            index=[name],
            data=[[r_squared, acc, mse, rmse, mae]], #mape
            columns=['R²', 'Precisão', 'MSE', 'RMSE', 'MAE'] #MAPE
            )
    return results


def Stats():
    # Header
    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1640886908/images/head_hub')
    st.title("Selecionar Entradas/Saídas")


    # Caminho do arquivo de dados pre-processados
    # data_path = './resources/data/spreadsheet.csv'
    # Widget de upload de arquivo
    excel_file = st.file_uploader("", type=['xlsx', 'xls'])

    # Para a execução se um arquivo não foi upado e não há dados préprocessados
    # if not excel_file and not os.path.exists(data_path):
    if not excel_file:
        st.stop()

    # Se for upado um arquivo, gere informações e faça pre-processamento
    if excel_file:
        with st.expander("Detalhes do arquivo"):
            st.write(f"**Nome do Arquivo:** {excel_file.name}")
            st.write(f"**Tipo do Arquivo:** {excel_file.type}")
            st.write(f"**Tamanho do Arquivo:** {excel_file.size / (1048576):.2f} MB")
        spreadsheet = helper.tratarPlanilha(excel_file)
        spreadsheet.to_csv("./resources/data/spreadsheet.csv", index=False)
        st.success("Upload e pré-processamento de dados realizado com sucesso!")

    ############################################################
    # filtra 0 < valores < 5 da camada superior
    maiorQ0 = spreadsheet['Camada_Superior'] > 0
    spreadsheet = spreadsheet[maiorQ0]
    menorQ5 = spreadsheet['Camada_Superior'] < 5
    spreadsheet = spreadsheet[menorQ5]

    # Seletor de outputs
    outputs = ['Camada_Superior']
    output_selector = st.multiselect("Saídas: ", spreadsheet.columns, default=outputs)

    # Seletor de inputs
    inputs = spreadsheet.columns.drop(output_selector)
    mat_filter = re.compile("Segmento_Prod_*")
    mat_list = list(filter(mat_filter.match, inputs))
    PARAMS = ['Camada_Superior' , 'Camada_Inferior', 'Vel_proc', 'Vel_Aplicador', 'Diametro_Aplicador', 'Vel_Alimentador', 'Visc_seg', 'Temp', 'Pressao_Aplicador', 'Pressao_Alimentador'] + mat_list
    for entry in output_selector:
        if entry in PARAMS: PARAMS.remove(entry)
    input_selector = st.multiselect("Entradas: ", inputs, default=PARAMS)

    # Garante a seleção de ao menos uma entrada e uma saída
    if not output_selector or not input_selector:
        st.warning("Selecione ao menos uma entrada e uma saída.")
        st.stop()

    # Gera DataFrames de entradas e saídas
    y = spreadsheet.loc[:, output_selector]
    X = spreadsheet.loc[:, input_selector]

    # Expander com preview das entradas
    with st.expander("Preview dos valores de Treinamento"):
        st.dataframe(X)

    # Slider para selecionar porcentagem de dados reservados para treinamento
    porc_treino = st.slider("Porcentagem dos dados reservada para treinamento:", 25, 90, 80)
    porc_treino /= 100

    # Hold out baseado na porcentagem definida
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=porc_treino, random_state =42)

    # Dicionário de modelos a serem utilizados
    MODELS = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.2),
            'Lasso Regression': Lasso(
                alpha=0.1,
                selection='random',
                max_iter=10000),
            'Random Forest Regressor': RandomForestRegressor(
                n_estimators=400,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                max_depth=250,
                bootstrap=False),
            'KNeighbors Regressor': KNeighborsRegressor(
                metric='manhattan',
                n_neighbors=50,
                weights='distance'),
            'Decision Tree Regressor': DecisionTreeRegressor(
                ccp_alpha=0.45,
                criterion='friedman_mse',
                max_features=None),
            'Bagging Regressor': BaggingRegressor(
                base_estimator=ExtraTreeRegressor(),
                n_estimators=20,
                bootstrap=True,
                bootstrap_features=False),
            'Elastic Net': ElasticNet(
                alpha=1.0,
                l1_ratio=1.0,
                selection='random',
                max_iter=10000),
            'MultiTask Elastic Net CV': MultiTaskElasticNetCV(
                l1_ratio=1,
                selection='cyclic',
                n_alphas=1000,
                max_iter=10000),
            'Extra Tree Regressor': ExtraTreeRegressor(
                max_depth=150,
                min_samples_leaf=6,
                min_samples_split=18,
                max_features='sqrt',
                ccp_alpha=0.265),
            'Gaussian Process Regressor': GaussianProcessRegressor(
                alpha=1e-4,
                kernel=DotProduct()),
            'Kernel Ridge': KernelRidge(
                kernel='poly',
                alpha=0.3,
                degree=2),
            'Multi Task Lasso CV': MultiTaskLassoCV(
                selection='cyclic',
                eps=1e-3,
                n_alphas=1000,
                max_iter=10000),
            'MLP Regressor': MLPRegressor(
                activation='relu',
                hidden_layer_sizes=(20, 20, 30),
                learning_rate_init=0.01,
                learning_rate='constant',
                alpha=0.001,
                max_iter=1000),
            'RANSAC Regressor': RANSACRegressor(
                base_estimator=KNeighborsRegressor(
                    metric='manhattan',
                    n_neighbors=50,
                    weights='distance'),
                min_samples=0.8,
                loss='absolute_error'),
            'Ridge CV': RidgeCV(alpha_per_target=False, gcv_mode='auto'),
            # 'Transformed Target Regressor': TransformedTargetRegressor(
            #     RandomForestRegressor(
            #         n_estimators=400,
            #         min_samples_split=4,
            #         min_samples_leaf=2,
            #         max_features='sqrt',
            #         max_depth=250,
            #         bootstrap=False),
            #     func=np.log1p,
            #     inverse_func=np.expm1
            #     )
            }

    # Medidor de porcentagem baseado na quantidade de modelos
    pct_done = 0
    pct_increase = 1/len(MODELS)

    # Inicializa DataFrame de resultados
    results_df = pd.DataFrame(columns=['R²', 'Precisão', 'MSE', 'RMSE', 'MAE'])

    # Usa spinner para indicar treinamento dos modelos
    with st.spinner("Treinando modelos..."):
        # Colunas para porcentagem e barra de progresso
        txt, bar = st.columns([1,9])
        # Inicializa barra de progresso e porcentagem
        progress_bar = bar.progress(0)
        pct_text = txt.empty()
        
        # Treina os modelos contidos em MODELS
        for name, mdl in MODELS.items():
            stats = gerarEstatisticas(mdl, name, X, y, porc_treino)
            # Armazena resultados em results_df
            results_df = pd.concat([results_df, stats])
            # Atualiza porcentagem
            pct_done += pct_increase
            pct_text.write(f"**{100*pct_done:.1f}%**")
            # Atualiza barra de progresso
            progress_bar.progress(pct_done)

        # Remove barra de progresso e porcentagem após terminar
        progress_bar.empty()
        pct_text.empty()

    ##################################################

    # Tabela de resultados
    st.subheader("Tabela com os melhores R² encontrados")
    st.table(results_df.sort_values(by=['R²'], ascending=False))

if __name__ == "__main__":
    Stats()
