import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config('Dataset Overview📊',layout='wide')
st.title('🚗 AI Driven CAR PRICE PREDICTION Platform')
st.subheader(":green[Smart Vehicle Price Estimation Using Machine Learning]",divider="yellow")
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
    st.info("""
    AI-powered system for intelligent vehicle price estimation and automobile market analysis.
    """)


df=pd.read_csv('Finaldf.csv')

col1,col2=st.columns(2)
with col1:
    st.info('Dataset Overview📊')
with col2:
    st.info(
    "Dataset used for analysis has been cleaned and preprocessed."
) 
col1,col2=st.columns(2)
with col1:
    st.write('No. of Rows :', df.shape[0])
with col2:
    st.write('No. of Columns :',df.shape[1])
with st.expander('📂 View Dataset Preview'):
    data=st.slider(":blue[Select Number Of Rows :]",5,len(df))
    st.dataframe(df.head(data))


tab1,tab2,tab3,tab4=st.tabs([":blue[Data types of columns]",':blue[Summary of Numerical columns]',":blue[Summary of Categorical Columns]",':blue[Unique values counts of Categorical Columns]'])

with tab1:
    st.write(df.dtypes)
with tab2:
    st.write(df.describe())
with tab3:
    st.write(df.describe(include='object'))
with tab4:
    columns=list(df.select_dtypes(include='object').columns)
    ana=st.selectbox("Select Coulmns ",columns)
    st.dataframe(df[ana].value_counts(),width=900,height=150)

col1,col2,col3,col4,col5=st.columns(5)

with col5:
     if st.button ('Analytics',icon=":material/arrow_forward:",type="primary"):
         st.switch_page('pages/Dashboard.py')    

    

