import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly import tools
import seaborn as sns

# numerical: discrete vs continuous

path = 'dataset.csv'


def load_data(path):
    data = pd.read_csv(path,sep = ',',index_col=0)
    return data


def plot_box(df,continuous):
    df_continuous = df[continuous]
    fig = px.box(df_continuous, x=df_continuous.columns,
                 
                 color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    st.plotly_chart(fig)

def count_plot(df):
    fig, ax = plt.subplots()
    ax = sns.countplot(df.Class.values,palette="husl")
    st.pyplot(fig)

def violinplot(df):
    fig, ax = plt.subplots()
    ax = sns.violinplot(df.Class.values,y=df.index.values,palette="husl")
    st.pyplot(fig)
    
def hist_plot(df,variable):
    fig, ax = plt.subplots()
    ax = sns.histplot(data = df,x = df[variable],kde=True,color='#1f77b4', )
    st.pyplot(fig)

def hist_hue(df,variable):
    fig, ax = plt.subplots()
    ax = sns.histplot(data = df,x = df[variable], kde=True, hue = df['Class'])
    st.pyplot(fig)

def scatter_plot(df,var1,var2):
    fig, ax = plt.subplots()
    ax = sns.scatterplot(data = df, x=var1, y=var2,hue = df['Class'])
    st.pyplot(fig)


def joint_plot(df,var1,var2):
    fig, ax = plt.subplots()
    ax = sns.jointplot(data = df, x=var1, y=var2, hue = df['Class'])
    st.pyplot(fig)

def pair_plot(df,var):
    df = df[var]
    fig, ax = plt.subplots()
    ax = sns.pairplot(df)
    st.pyplot(fig)

def plot_corr(df):
    features = df.columns.values[0:10]
    corr = df[features].corr().round(2).to_numpy().tolist()
    fig = ff.create_annotated_heatmap(corr,x = features.tolist(), y = features.tolist(),showscale=True,
                zmin=0, zmax=1,
               )

    fig.update_layout(width=800, height=700,  template="plotly_dark")

    fig.update_yaxes(
        autorange="reversed"
    )    

    # Make text size bigger
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 14

    return st.plotly_chart(fig)
    
    


def get_data():
    df = load_data(path)


    st.title('Data')
   
    

    st.markdown('### DataFrame `dataset`')
    st.markdown('Dataset shape: {}'.format(df.shape))
    st.markdown('253 entries  |  154 columns')
    st.write(df)
 
    
    st.markdown('### DataFrame  `dataset description`')
    st.markdown('Dataset shape: {}'.format(df.shape))
    st.markdown('253 entries  |  154 columns')
    st.write(df.describe().T)

def get_data_description():
    df = load_data(path)
    discrete = [var for var in df.columns if df[var].dtype!='O' and var!='Class' and df[var].nunique()<10]
    continuous = [var for var in df.columns if df[var].dtype!='O' and var!='Class' and var not in discrete]

    st.title('Feature Description')


    menu_variables= st.radio(
        "",
        ("Intro variables", "Class","X0", "X47", "X141", "X147"),
    )

    if menu_variables == "Intro variables":
        col1, col2 = st.columns(2)

        with col1:
        
            st.markdown('### Continuous variables')
            st.markdown('There are {} continuous variables'.format(len(continuous)))
            st.markdown(' ')
            st.write(plot_box(df,continuous[:20]))

        with col2:
            st.markdown('### Discrete variables')
            st.markdown('There are {} discrete variables'.format(len(discrete)))
            st.markdown(' ')
            st.write(plot_box(df,discrete[:20]))
       
        
    elif menu_variables == "Class":
        with st.expander("Description of Class variable"):
         st.code(df.Class.describe())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### Class counts')
            st.markdown(' ')
            count_plot(df)
        with col2:
            st.markdown('### Class distribution')
            st.markdown(' ')
            violinplot(df)
    elif menu_variables == "X0":
        with st.expander("Description of X0 variable"):
         st.code(df.X0.describe())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### X0 counts')
            st.markdown(' ')
            hist_plot(df,'X0')
        with col2:
            st.markdown('### X0 distribution')
            st.markdown(' ')
            hist_hue(df,'X0')
        # variables_metascore(movies)
    elif menu_variables == "X47":
        with st.expander("Description of X47 variable"):
         st.code(df.X47.describe())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### X47 counts')
            st.markdown(' ')
            hist_plot(df,'X47')
        with col2:
            st.markdown('### X47 distribution')
            st.markdown(' ')
            hist_hue(df,'X47')
        # variables_budget(movies)
    elif menu_variables == "X141":
        with st.expander("Description of X141 variable"):
         st.code(df.X141.describe())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### X141 counts')
            st.markdown(' ')
            hist_plot(df,'X141')
        with col2:
            st.markdown('### X141 distribution')
            st.markdown(' ')
            hist_hue(df,'X141')
        # variables_gross(movies)
    elif menu_variables == "X147":
        with st.expander("Description of X147 variable"):
         st.code(df.X147.describe())
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('### X147 counts')
            st.markdown(' ')
            hist_plot(df,'X147')
        with col2:
            st.markdown('### X147 distribution')
            st.markdown(' ')
            hist_hue(df,'X147')

def get_relationships():
    df = load_data(path)
    st.title('Relationship between variables')

    menu_relations= st.radio(
        "",
        ("X0/X47", "X141/X147", "X0/147", "X47/X141"),
    )

    if menu_relations == "X0/X47":
        st.markdown('### Relationship between X0 and X47')
        scatter_plot(df, 'X47', 'X0')

        

    elif menu_relations == "X141/X147":
        st.markdown('### Relationship between X141 and X147')
        scatter_plot(df, 'X147', 'X141')

        
    elif menu_relations == "X0/147":
        st.markdown('### Relationship between X0 and X147')
        scatter_plot(df, 'X147', 'X0')

      

    elif menu_relations == "X47/X141":
        st.markdown('### Relationship between X47 and X141')
        scatter_plot(df, 'X47', 'X141')


def get_correlation_matrices():
    df = load_data(path)
    st.title('Correlation matrices')
    plot_corr(df)




      
   