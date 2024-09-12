# import os
import re
import pandas as pd
import streamlit as st
import resources.lib.helper as helper


def Equacao():
    # Header

    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1640886908/images/head_hub')
    st.title("Equação")

    # Widget de upload de arquivo
    excel_file = st.file_uploader("", type=['xlsx', 'xls','csv'])

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

        # filtra 0 < valores < 5 da camada superior
        maiorQ0 = spreadsheet['Camada_Superior'] > 0
        spreadsheet = spreadsheet[maiorQ0]
        menorQ5 = spreadsheet['Camada_Superior'] < 5
        spreadsheet = spreadsheet[menorQ5]

    # Caminho do arquivo de dados pre-processados
    # data_path = './resources/data/spreadsheet.csv'
    #
    # # Checa se existe arquivo pre-processado e o carrega como DataFrame caso exista
    # if os.path.exists(data_path):
    #     spreadsheet = pd.read_csv(data_path)
    # else:
    #     st.warning("##### É necessário fazer o upload e pre-processamento da planilha antes de proceder.")
    #     st.stop()

    ############################################################

    # Subtitle
    st.write("#### Descarte de variáveis e agrupamento de dados")
    
    # Inicializa colunas
    l, r = st.columns(2)



    # Seletor de outputs
    output = ['Camada_Superior']
    output_selector = st.multiselect("Selecione apenas uma saída: ", spreadsheet.columns, default=output)

    # Seletor de inputs
    inputs = spreadsheet.columns.drop(output_selector)
    mat_filter = re.compile("Segmento_Prod_*")
    mat_list = list(filter(mat_filter.match, inputs))
    PARAMS = ['Camada_Superior', 'Camada_Inferior', 'Vel_proc', 'Vel_Aplicador', 'Diametro_Aplicador',
              'Vel_Alimentador', 'Visc_seg', 'Temp', 'Pressao_Aplicador', 'Pressao_Alimentador'] + mat_list
    for entry in output_selector:
        if entry in PARAMS: PARAMS.remove(entry)
    input_selector = st.multiselect("Entradas: ", inputs, default=PARAMS)

    # Garante a seleção de ao menos uma entrada e uma saída
    if not output_selector or not input_selector:
        st.warning("Selecione ao menos uma entrada e uma saída.")
        st.stop()



    # Fit dos modelos
    models = helper.gerarEquacao(input_selector, output, spreadsheet)

    # Subtitle
    st.write("### Tabelas de coeficientes")

    # Inicializa colunas
    cols = st.columns(3)
    # Coloca as variáveis e coeficientes em tabelas
    for ind, mdl in enumerate(models):
        col = cols[ind%3]
        col.write(f"### Modelo {mdl}")
        col.write(f"### **R²:** {models[mdl].rsquared:.3f}")
        coeff_table = pd.DataFrame(models[mdl].params, columns=['Coeficiente'])
        col.table(coeff_table)

    # Expanders com sumários dos modelos
    with st.expander("Sumários detalhados dos modelos"):
        for mdl in models:
            st.write(f"### Sumário detalhado do modelo {mdl}")
            st.write(models[mdl].summary())
            st.write("\n\n")

    # Gerar string com equações
    equations = {}
    for model in models:
        coeffs = [ round(coeff, 5) for coeff in models[model].params ]
        params = input_selector
        equations[model] = f"##### {output_selector} = {coeffs[0]}  <font size=4>+</font>  " + "  <font size=4>+</font>  ".join([ f"({coeff}) × <font size=5; color=#FF4B4B>{param}</font>" for coeff, param in zip(coeffs[1:], params) ])

    # Encontrar modelo de melhor R^2
    best_r2_model = 'OLS'
    for model in models:
        if models[model].rsquared > models[best_r2_model].rsquared:
            best_r2_model = model

    # Mostrar equação do melhor modelo
    st.write(f"### Equação do modelo {best_r2_model}")
    st.markdown(equations[best_r2_model],unsafe_allow_html=True)

    # Mostrar equações dos demais modelos em expanders
    for model in models:
        if model != best_r2_model:
            with st.expander(f"Equação do modelo {model}:"):
                st.markdown(equations[model], unsafe_allow_html=True)

    # st.dataframe(selected_espec)
    # i = st.number_input("", 5)
    # st.write(f"Expect: {selected_espec['Média Revestimento'][i]}")
    # st.write(models['OLS'].predict([1]+ list(selected_espec[input_selector].iloc[i].values)))

if __name__ == "__main__":
    Equacao()
