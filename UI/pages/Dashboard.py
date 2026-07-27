import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

st.set_page_config(page_title='CAR DATASET VISUALIZATION',
                   page_icon="🚗",
                   layout='wide')
st.markdown("""
<style>

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}

[data-testid="stSidebar"] * {
    color: white;
}

</style>
""", unsafe_allow_html=True)
with st.sidebar:

    st.title("🚗 Car Price AI")


    st.page_link(
        "DataSet_OverView.py",
        label="🏠 Dataset Overview"
    )

    st.page_link(
        "pages/Dashboard.py",
        label="📊 Analytics"
    )

    st.page_link(
        "pages/Prediction.py",
        label="💰 Prediction"
    )
    st.info(''' 
        Performed univariate, bivariate, and multivariate analysis

        Analyzed numerical features using box plots and scatter plots

        Explored categorical features using pie charts and bar charts

        Identified distributions, patterns, relationships, and outliers in the dataset

        Created interactive visualizations for better data interpretation using Plotly

        ''')


st.title(':blue[Car] :blue[Dataset] :gray[Visualization]👁️‍🗨️')
st.subheader(':red[Analytics!]',divider='red')

df=pd.read_csv('Finaldf.csv')
num_cols1 = df.select_dtypes(include = "number").columns.tolist()
num_cols2 = df.select_dtypes(include = "number").columns.tolist()
cat_cols = df.select_dtypes(include = "object").columns.tolist()
st.subheader(":blue[Choose your chart and columns]")


chart_type = st.selectbox(":rainbow[Select Chart Type]",["Bar Chart","Scatter Plot","Box Plot","Count Plot"])






if chart_type=='Scatter Plot':
    st.subheader(':orange[Scatter Plot]')
    x_col = st.selectbox(":rainbow[Select numerical column (x-axis)]",num_cols1)
    y_col = st.selectbox(":rainbow[Select numerical column (Y-axis)]",num_cols2)
    hue=st.selectbox(":rainbow[Select Categorical column (hue)]",cat_cols)
    
    if x_col==y_col:
        st.warning('⚠️Select different numerical column on Y-axis')
    elif st.button(':rainbow[Create chart]'):
        fig=px.scatter(df,x_col,y_col,
                       title=f'{x_col} vs {x_col} vs {hue} Distribution',
                       color=hue)
        st.plotly_chart(fig)

if chart_type=='Bar Chart':
    st.subheader(':violet[Bar Plot]')
    x_col = st.selectbox(":rainbow[Select Categorical column (x-axis)]",cat_cols)
    y_col = st.selectbox(":rainbow[Select Numerical column (Y-axis)]",num_cols1)
    if st.button(':rainbow[Create chart]'):
        fig=px.bar(df
                 ,x_col,y_col,
                title=f'{x_col} vs {y_col} Distribution',
                width=100,height=600)
        st.plotly_chart(fig)   

if chart_type=='Count Plot':
    st.subheader(":green[Count Chart]")
    val = st.selectbox(":rainbow[Select Categorical column (x-axis)]",cat_cols)
    if st.button(':rainbow[Create chart]'):
        fig=px.histogram(df,x=val,color=val,
                         title=f'{val} Distribution')
        st.plotly_chart(fig)
        
        st.pyplot(plt)
if chart_type=='Box Plot':
    st.subheader(':gray[Box Chart]')
    
    y_col=st.selectbox(":rainbow[Select Categoricsl column (Y-axis)]",num_cols1)  
    if st.button(':rainbow[Create chart]'):
        fig=px.box(df,y=y_col,width=50,height=600,
                 title=f'{y_col} Distribution')
        st.plotly_chart(fig) 


col1,col2,col3,col4,col5=st.columns(5)

with col1:
    if st.button("Data Overview",icon=":material/arrow_back:",type="primary"):
        st.switch_page('DataSet_OverView.py')
with col5:
    if st.button(' Prediction',icon=":material/arrow_forward:",type='primary'):
        st.switch_page('pages/Prediction.py')
        
                


