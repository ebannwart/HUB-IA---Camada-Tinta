import streamlit as st
import resources.lib.pageEq as page_eq
import resources.lib.pageAbout as page_about
import resources.lib.pageStats as page_stats
import resources.lib.pageCalc as page_calculadora
# import resources.lib.pageUpload as page_upload

st.set_page_config(
     page_title="CSN",
     page_icon="https://www.guide.com.br/wp-content/uploads/2020/07/logo-csn-escudo-256.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "# This is a header. This is an *extremely* cool app!"
     }
 )

############################################################

def main():
    st.sidebar.image(r'C:\Users\cs64280\OneDrive - Companhia Siderurgica Nacional\CTM\Projetos\Sprint 4 - HubInovação\Entrega Sprint4\Codigos\app_streamlit\resources\imageslogo-csn-nome.png')
    

    menu_dict = {
            #"Upload de dados": page_upload.Upload,
            "Equação": page_eq.Equacao,
            "Avaliar Modelos": page_stats.Stats,
            "Calculadora": page_calculadora.Calculadora,
            "About": page_about.About
            }

    choice = st.sidebar.radio("Menu", menu_dict.keys())

    menu_dict[choice]()


if __name__ == '__main__':
    main()