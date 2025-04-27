import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import streamlit.components.v1 as components
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot, plot
import sys, os
import utils.textos as tx
import base64
import pickle

# configuración página
def config_page():
    st.set_page_config(page_title = 'Trabajo Fin Grado', layout = 'wide')
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

def mostrar(path, width):
    img = Image.open(path)
    st.image(img, width=width)

def cargar_datos(path):
    df = pd.read_csv(path)
    return df

def generar(orden, df):
    sns.barplot(x=orden,y='Model',data=df,color='b')
    plt.title('Model Compare Graphic');
    plt.savefig("utils/image/comp.jpg")

def portada():
    
    st.markdown('##')

    mostrar('utils/image/apo.jpeg', 1000)
    st.markdown('##')
    st.markdown('##')

    with st.expander('RESUMEN'):
        st.write(tx.resumen)
    
    with st.expander('DATASET'):
        st.write(tx.dataset)

def predecir():

    with open('data/encoder', "rb") as f:
        oHec = pickle.load(f)
    with open('data/scaler', "rb") as f:
        scaler = pickle.load(f)
    with open('model/modelo', 'rb') as file:  
        model = pickle.load(file)
    
    st.markdown('##')
    st.markdown('##')
    st.markdown("<h5 style='text-align: left;'>Responda las siguientes preguntas:</h5>", unsafe_allow_html=True)
    st.markdown('##')
    genre = st.radio("¿Cuál es su género?", ('Hombre', 'Mujer', 'Otro'))
    st.markdown('##')
    age = st.number_input('Inserte edad', min_value=0, max_value=150, step=1)
    st.markdown('##')
    tension = st.radio("¿Padece hipertensión?", ('Sí', 'No'))
    st.markdown('##')
    cardio = st.radio("¿Padece alguna enfermedad cardiovascular?", ('Sí', 'No'))
    st.markdown('##')
    casado = st.radio("¿Alguna vez se ha casdado?", ('Sí', 'No'))
    st.markdown('##')
    work = st.radio("¿Qué tipo de trabajo tiene?", ('Nunca he trabajado', 'Demasiado joven para trabajar', 'Empresa pública', 'Autónomo', 'Trabajo por cuenta ajena'))
    st.markdown('##')
    casa = st.radio("¿Qué tipo de residencia tiene?", ('Rural', 'Urbana'))
    st.markdown('##')
    glucosa = st.number_input("¿Cuál es su nivel medio de glucosa en sangre?", min_value=0.0, max_value=300.0, step=0.1)
    st.markdown('##')
    imc = st.number_input("¿Cuál es su índice de masa corporal?", min_value=0.0, max_value=100.0, step=0.1)
    st.markdown('##')
    fuma = st.radio("¿Se considera fumador?", ('He fumado', 'Nunca he fumado', 'Fumo', 'Otro'))
    st.markdown('##')

    col1, col2, col3 , col4, col5 = st.columns(5)
    with col3 :
        pred_bt = st.button('Predecir', type="secondary")

    if pred_bt:

        numerical_features = []
        binary_features = []
        categorical_features = []

        if genre == 'Hombre':
            binary_features.append(0)
        elif genre == 'Mujer':
            binary_features.append(1)
        else:
            binary_features.append(2)
        
        numerical_features.append(age)

        if tension == 'Sí':
            binary_features.append(1)
        else:
            binary_features.append(0)

        if cardio == 'Sí':
            binary_features.append(1)
        else:
            binary_features.append(0)

        if casado == 'Sí':
            binary_features.append(1)
        else:
            binary_features.append(0)

        if work == 'Nunca he trabajado':
            categorical_features.append('Never_worked')
        elif work == 'Demasiado joven para trabajar':
            categorical_features.append('children')
        elif work == 'Empresa pública':
            categorical_features.append('Govt_job')
        elif work == 'Autónomo':
            categorical_features.append('Self-employed')
        else:
            categorical_features.append('Private')

        if casa == 'Urbana':
            binary_features.append(1)
        else:
            binary_features.append(0)

        numerical_features.append(glucosa)
        numerical_features.append(imc)

        if fuma == 'He fumado':
            categorical_features.append('formerly smoked')
        elif fuma  == 'Nunca he fumado':
            categorical_features.append('never smoked')
        elif fuma == 'Fumo':
            categorical_features.append('smokes')
        else:
            categorical_features.append('Unknown')

        cat_df = pd.DataFrame([categorical_features], columns = ['work_type', 'smoking_status'])
        cat_df = oHec.transform(cat_df)
        cat_df = pd.DataFrame(cat_df.toarray(), columns=oHec.get_feature_names_out())
        num_df = pd.DataFrame([numerical_features], columns = ['age', 'avg_glucose_level', 'bmi'])
        num_df = scaler.transform(num_df)
        num_df = pd.DataFrame(num_df, columns=['age', 'avg_glucose_level', 'bmi'])
        bin_df = pd.DataFrame([binary_features], columns = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type'])
        X = pd.concat([num_df, bin_df, cat_df], axis=1)
        # pred = model.predict(X)
        # st.markdown('##')
        # # Predicción binaria
        # if pred[0] == 0:
        #     st.markdown("<h3 style='text-align: left;'>Negativo, no hay riesgo de apoplejía</h3>", unsafe_allow_html=True)
        # else:
        #     st.markdown("<h3 style='text-align: left; color: red;'>Positivo, alta probabilidad de ictus</h3>", unsafe_allow_html=True)
        prop_pred = model.predict_proba(X)
        st.markdown('##')
        # Probabilidad de padecer ictus
        if prop_pred[0][1] < 0.33:
            st.markdown(f"<h3 style='text-align: left; color: black;'>Riesgo bajo de padecer ictus</h3>", unsafe_allow_html=True)
        elif prop_pred[0][1] < 0.66:
            st.markdown(f"<h3 style='text-align: left; color: orange;'>Riesgo medio de padecer ictus</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: left; color: red;'>Riesgo alto de padecer ictus</h3>", unsafe_allow_html=True)


def conclusiones():

    st.markdown('##')
    with st.expander('1ª CONCLUSIÓN'):
        st.write(tx.con1)
    
    with st.expander('2ª CONCLUSIÓN'):
        st.write(tx.con2)