import streamlit as st
import utils.funciones as fc

# configurar página
fc.config_page()

st.title('Predicción Ataque Cerebrovascular')

# menu
menu = st.sidebar.selectbox('Menú', ['Portada', 'Predicción', 'Conclusiones'])

if menu == 'Portada':
    fc.portada()

elif menu == 'Predicción':
    fc.predecir()

else:
    fc.conclusiones()