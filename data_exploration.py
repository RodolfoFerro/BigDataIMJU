# ==============================================================
# Author: Rodoflo Ferro
# Twitter: @FerroRodolfo
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script has been originally created by Rodolfo Ferro.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ==============================================================

# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np


# Load data
data = pd.read_csv('data/datos_ventas.csv')

# Intro section
st.title("Exploración de datos y preprocesamiento")
st.write(
    """
    Bienvenid@ a este sencillo ejemplo que ejecuta algunos procesos previos
    al modelado de la información. Dichos procesos involucran transformaciones,
    limpieza y llenado de datos faltantes para tener un conjunto de datos que
    pueda ser modelado.
    """
)

# Data loading
st.header("Cargando los datos")
st.markdown(
    """
    A continuación cargamos los datos (originales) a utilizar. Dichos datos pueden ser
    encontrados en la carpeta de datos como `data/datos_ventas.csv`.

    > **Nota:** Los datos y el código fuente de esta aplicación web pueden ser encontrados en
    > el repositorio [https://github.com/RodolfoFerro/BigDataIMJU](https://github.com/RodolfoFerro/BigDataIMJU).

    Los datos pueden ser cargados utilizando pandas con las líneas:
    ```python
    import pandas as pd

    data = pd.read_csv('data(datos_ventas.csv')
    ```

    Al desplegar los datos deberías ver una tabla como la siguiente:
    """
)
st.dataframe(data)

# Grab dependent and independent variables
st.header("Datos dependientes e independientes")
st.markdown(
    """
    Podemos separar los datos de la tabla en variables dependientes e independientes.

    Los datos pueden ser cargados utilizando pandas con las líneas:
    ```python
    import pandas as pd

    data = pd.read_csv('data(datos_ventas.csv')
    ```

    Al desplegar los datos deberías ver una tabla como la siguiente:
    """
)
st.dataframe(data)