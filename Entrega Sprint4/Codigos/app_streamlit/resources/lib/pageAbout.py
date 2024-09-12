import streamlit as st
import streamlit.components.v1 as components

def About():
    st.subheader("Residencia Hub IA/ Fiep / SENAI-PR")
    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1640886909/images/banner_hub.jpg', output_format="auto")
    components.html("""
            <div style="text-align: center;"><a href="https://www.senaipr.org.br/tecnologiaeinovacao/nossarede/hubia/"><img src="https://cdn3.iconfinder.com/data/icons/social-media-circle-long-shadow/1024/longShadow-256.png" alt="" width="36" height="36" /> </a> 
            <a href="https://br.linkedin.com/company/hubiasenai"><img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-512.png" alt="" width="36" height="36" /> </a> 
            </div>
            """)
    ############################################################
    # st.subheader("Residente Andre Mikio")
    st.info("Gonçalves, Afrânio")
    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1652707400/hub_ia_senai_fase_05_CSN/residente_afranio_goncalves.jpg', output_format="auto")
    components.html("""
            <div style="text-align: center;">
            <a href="https://www.linkedin.com/in/afr%C3%A2nio-gon%C3%A7alves-872b7a50"><img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-512.png" alt="" width="36" height="36" /> </a> 
               """)
    ############################################################
    # st.subheader("Residente Luiz Brandão (Soldado)")
    # index.carrega('Files/Code/index.html')
    # st.info("Built with Streamlit")
    st.info("Brandão, Luiz @uflaniano")
    st.image('https://res.cloudinary.com/dmbamuk26/image/upload/v1640886909/images/banner_brandao.jpg', output_format="auto")
    components.html("""
            <div style="text-align: center;">
            <a href="https://www.uflaniano.com.br/"><img src="https://cdn3.iconfinder.com/data/icons/social-media-circle-long-shadow/1024/longShadow-256.png" alt="" width="36" height="36" /> </a> 
            <a href="http://buscatextual.cnpq.br/buscatextual/visualizacv.do?id=K4817683Y6"><img src="https://cdn2.iconfinder.com/data/icons/bitsies/128/EditDocument-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://github.com/zolpy"><img src="https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_github_2-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://www.linkedin.com/in/luiz-carlos-brand%C3%A3o-junior-605b94b1/"><img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Linkedin_unofficial_colored_svg-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://twitter.com/uflaniano"><img src="https://cdn2.iconfinder.com/data/icons/social-media-2285/512/1_Twitter3_colored_svg-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://www.facebook.com/uflaniano"><img src="https://cdn1.iconfinder.com/data/icons/logotypes/32/square-facebook-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://www.instagram.com/uflaniano/"><img src="https://cdn2.iconfinder.com/data/icons/social-media-applications/64/social_media_applications_3-instagram-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://br.pinterest.com/uflaniano/"><img src="https://cdn2.iconfinder.com/data/icons/2018-social-media-app-logos/1000/2018_social_media_popular_app_logo_pinterest-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://www.tiktok.com/@uflaniano"><img src="https://cdn4.iconfinder.com/data/icons/logos-brands-in-colors/2840/tiktok-logo-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://www.youtube.com/channel/UCc6XH0e8US45u5IkoF6Ul0A"><img src="https://cdn4.iconfinder.com/data/icons/socialcones/508/YouTube-512.png" alt="" width="36" height="36" /> </a> 
            <a href="https://hub.docker.com/u/uflaniano"><img src="https://cdn3.iconfinder.com/data/icons/social-media-2169/24/social_media_social_media_logo_docker-512.png" alt="" width="36" height="36" /> </a></div>
            """)
