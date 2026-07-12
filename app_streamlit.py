import streamlit as st
import pandas as pd

st.set_page_config(page_title="Mi App Streamlit", layout="wide")

st.title("Bienvenido a mi aplicación Streamlit")

st.write("Esta es una plantilla básica para empezar a desarrollar.")

# Agrega más elementos aquí (gráficos, tablas, etc.)
# Ejemplo de datos
data = pd.DataFrame({
    'Columna A': [1, 2, 3],
    'Columna B': [10, 20, 30]
})

st.write("Ejemplo de tabla de datos:")
st.dataframe(data)

st.sidebar.header("Opciones")
opcion = st.sidebar.selectbox("Selecciona una opción", ["Opción 1", "Opción 2"])

st.write(f"Has seleccionado: {opcion}")
