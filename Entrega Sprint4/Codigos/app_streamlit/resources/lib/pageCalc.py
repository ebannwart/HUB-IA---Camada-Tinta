# import os
import re
# import numpy as np
import pandas as pd
import streamlit as st
# from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, MultiTaskElasticNetCV, MultiTaskLassoCV, RANSACRegressor, Ridge, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
import resources.lib.helper as helper
import pickle
from joblib import dump, load
# dump(reg, 'regression_model_saved.joblib')
# reg_loaded = load('regression_model_saved.joblib')

def Calculadora():
    # Header
    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1640886908/images/head_hub')
    st.title("Calculadora")

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
        # spreadsheet.to_csv("./resources/data/spreadsheet.csv", index=False)
        st.success("Upload e pré-processamento de dados realizado com sucesso!")

    # excel_file = st.file_uploader("", type=['xlsx', 'xls'])

    # Para a execução se um arquivo não foi upado e não há dados préprocessados
    # if not excel_file and not os.path.exists(data_path):
    # if not excel_file:
    #     st.stop()
    #
    # # Se for upado um arquivo, gere informações e faça pre-processamento
    # if excel_file:
    #     with st.expander("Detalhes do arquivo"):
    #         st.write(f"**Nome do Arquivo:** {excel_file.name}")
    #         st.write(f"**Tipo do Arquivo:** {excel_file.type}")
    #         st.write(f"**Tamanho do Arquivo:** {excel_file.size / (1048576):.2f} MB")
    #     spreadsheet = helper.tratarPlanilha(excel_file)
    #     # spreadsheet.to_csv("./resources/data/spreadsheet.csv", index=False)
    #     st.success("Upload e pré-processamento de dados realizado com sucesso!")

    # Caminho do arquivo de dados pre-processados
    # data_path = './resources/data/spreadsheet.csv'
    #
    # # Checa se existe arquivo pre-processado e o carrega como DataFrame caso exista
    # if os.path.exists(data_path):
    #     spreadsheet = pd.read_csv(data_path)
    # else:
    #     st.warning("##### É necessário fazer o upload e pre-processamento da planilha antes de proceder.")
    #     spreadsheet = pd.DataFrame()
    #     st.stop()

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

    # Slider para selecionar porcentagem de dados reservados para treinamento
    porc_treino = st.slider("Porcentagem dos dados reservada para treinamento:", 25, 90, 80)
    porc_treino /= 100

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
            #     func=np.log,
            #     inverse_func=np.exp
            #     )
            }

    # Subheader
    st.subheader("Valores de entrada")

    # Entradas dinâmicas baseadas na lista de inputs
    entries = []
    # Inicializa colunas
    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]
    for ind, entrada in enumerate(input_selector):
        col = cols[ind%(len(cols))]
        # Usa moda da entrada como valor inicial
        default_value = float(spreadsheet[entrada].mode()[0])
        # Parâmetros padrão do widget
        min_value, max_value, increment =  None, None, None
        # Altera parâmetros no caso de Materiais
        if entrada in mat_list:
            default_value = int(default_value)
            min_value, max_value, increment = 0, 1, 1
        # Cria widget
        entry = col.number_input(
                label=entrada,
                key=entrada,
                value=default_value,
                min_value=min_value,
                max_value=max_value,
                step=increment)
        # Armazena resultados na lista entries
        entries.append(entry)

    # Modelo selecionado por padrão
    default_model = 'Random Forest Regressor'

    # Seletor de modelos
    model_selection = st.selectbox("Selecione o modelo:", MODELS.keys(), index=list(MODELS.keys()).index(default_model))

    # Gera DataFrames de entradas e saídas
    y = spreadsheet.loc[:, output_selector]
    X = spreadsheet.loc[:, input_selector]

    # Hold out baseado na porcentagem definida
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=porc_treino, random_state=42)

    # Inicializa o modelo selecionado
    prediction_model = MODELS[model_selection]

    # Fit do modelo
    prediction_model.fit(X_train, y_train)

    st.write(f"O modelo selecionado tem um **R²** de **{prediction_model.score(X_test, y_test):.4f}**")

    dump(prediction_model, './resources/model/' +model_selection + '.joblib')
    # with open('./resources/model/' +model_selection + '.pkl', 'wb') as files:
    #     pickle.dump(prediction_model, files)


    # Aviso sobre o modelo selecionado por padrão
    # st.info(f"O modelo selecionado por padrão ({default_model}) é o modelo recomendado, trazendo os resultados mais consistentes.")

    # Esconder índice na tabela e alinhar à esquerda
    hide_table_row_index = """
                <style>
                    tbody th {display:none}
                    .blank {display:none}
                    .col_heading {text-align: left !important}
                    td {text-align: left !important}
                </style>
                """
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    # Utiliza o modelo para prever os valores de saída
    entry_table = pd.DataFrame(data=[entries], columns=input_selector)
    predictions = pd.DataFrame(prediction_model.predict(entry_table), columns=output_selector)
    # Exibe resultados
    st.write("#### Valores previstos pelo modelo")
    st.table(predictions.style.format("{:.2f}"))

if __name__ == "__main__":
    Calculadora()
