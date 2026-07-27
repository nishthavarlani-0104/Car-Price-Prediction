import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

model=joblib.load('final_model.pkl')
le=joblib.load('le.pkl')
sc=joblib.load('sc.pkl')
cmap=joblib.load('condition_map.pkl')


st.set_page_config(page_title='CAR PRICE PREDICTION',
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
    st.success("Accuracy : 82%")
    
    with st.expander("Feature and Target Info"):
        st.markdown("""
                Final selected features:
                - Year
                - Manufacturer
                - Model
                - Title Status
                - Cylinders
                - Fuel Type
                - Odometer
                - Transmission
                - Drive Type
                - Vehicle Type
                    
                Target:
                - Price
                                            
                    """)
        
df=pd.read_csv('Finaldf.csv')
condition_order = [
    "new",
    "like new",
    "excellent",
    "good",
    "fair",
    "salvage"]        
    

manufacturer_le = LabelEncoder()
model_le = LabelEncoder()
cylinders_le = LabelEncoder()
fuel_le = LabelEncoder()
transmission_le = LabelEncoder()
drive_le = LabelEncoder()
type_le = LabelEncoder()
title_le=LabelEncoder()

# ---------------- FIT LABEL ENCODERS ----------------

manufacturer_le.fit(df['manufacturer'])

model_le.fit(df['model'])

cylinders_le.fit(df['cylinders'])

fuel_le.fit(df['fuel'])

transmission_le.fit(df['transmission'])

drive_le.fit(df['drive'])

type_le.fit(df['type'])    

title_le.fit(df['title_status'])


st.title('🤖 :blue[Car Price Prediction]')
st.badge('Model Selected : XGBoost',color='grey',icon=':material/star:')
st.divider()

col1, col2 = st.columns(2)

# ---------------- COLUMN 1 ----------------
with col1:

    st.markdown("""
    ### ⚙️ How To Use

    1. Enter vehicle details  
    2. Select required input features  
    3. Click on **Predict Price**  
    4. Get instant vehicle price estimation  
    """)

# ---------------- COLUMN 2 ----------------
with col2:

    st.markdown("""
    ### 🧠 Feature Importance & Selection

    Initially, the dataset contained 13 features.  
    After performing feature importance analysis and feature selection, the final model was trained using 10 important features for better efficiency and prediction performance.
    """)
st.divider()
st.subheader(":red[Enter the values properly]✅")  


col1,col2=st.columns(2)
with col1:
    year=st.number_input('Enter Manufacture Year of Car',min_value=1990,max_value=2026,step=1)
with col2:
    odometer=st.number_input('Enter Odometer Reading',min_value=0,max_value=500000,step=1000)


col1,col2,col3=st.columns(3)

with col1:
    manufacturer=st.selectbox("Select the Manufacturer of Car",df.manufacturer.unique())
with col2:
    model_name=st.selectbox('Select the model of Car',df.model.unique())
with col3:
    title=st.selectbox('Select the status of Car',df.title_status.unique())

col4,col5,col6=st.columns(3)

with col4:
    cylinders=st.selectbox('Select Cylinders of Car',df.cylinders.unique())       
with col5:
    fuel=st.selectbox('Select Fuel Type of Car',df.fuel.unique())
with col6:
    transmission=st.selectbox('Select transmission type of Car',df.transmission.unique())

col7,col8=st.columns(2)
with col7:
    drive=st.selectbox ('Select Drive Type of Car',df.drive.unique())
with col8:
    vehicle=st.selectbox('Select Vehicle Type of Car',df.type.unique())  


manufacturer = manufacturer_le.transform([manufacturer])[0]
model_name = model_le.transform([model_name])[0]
cylinders = cylinders_le.transform([cylinders])[0]
fuel = fuel_le.transform([fuel])[0]
transmission = transmission_le.transform([transmission])[0]
drive = drive_le.transform([drive])[0]
vehicle = type_le.transform([vehicle])[0]
title=title_le.transform([title])[0]

input_df = pd.DataFrame({
    'year': [year],
    'manufacturer': [manufacturer],
    'model': [model_name],
    'cylinders': [cylinders],
    'fuel': [fuel],
    'odometer': [odometer],
    'title_status': [title],
    'transmission': [transmission],
    'drive': [drive],
    'type': [vehicle]
})

input_df[['year','odometer']]=sc.transform(input_df[['year','odometer']])

if st.button(' Predict Price',type='primary'):
    prediction=model.predict(input_df)[0]
    col1,col2=st.columns(2)
    with col1:
        st.success(f'Estimated Car Price💰  {prediction:,.2f}')
    with col2:
            st.warning(
    "Model R² Score :"
    "82%"
)
            
col1,col2=st.columns(2)
with col1:
    if st.button(' Dashboard',icon=":material/arrow_back:",type='primary'):
        st.switch_page('pages/Dashboard.py')            