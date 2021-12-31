import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import *

st.sidebar.image('image.jpeg', width=200)

st.sidebar.header('EDA Tools')
st.sidebar.markdown('Exploratory data analysis')


menu = st.sidebar.radio(
    "",
    ("Data", "feature decription", "Relationships between features", "correlation matrices"),
)

st.sidebar.markdown('---')
st.sidebar.write('Aregbede Victor | December 2021 aregbede60@gmail.com')

if menu == 'Data':
    get_data()

elif menu == 'feature decription':
    get_data_description()

elif menu == 'Relationships between features':
    get_relationships()

elif menu == 'correlation matrices':
    get_correlation_matrices()