import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import ydata_profiling  
import pandas as pd
import os
# import pickle


if os.path.exists('./source_data.csv'):
    df = pd.read_csv('source_data.csv', index_col=None)

with st.sidebar:
    st.image("8a0039b8be78dfc26b21b77a0fe0fba6b00863887029b7f3f38f83d7.jpg")
    st.title("AutpStreamML")
    choice = st.radio("Navigation" , ["upload" , "Profiling" , "ML" , "Download"])
    st.info("This applicationallows you to build automated ML applications pipleine using Streamlit , Pnadas Profiling and PyCaret")


if choice == 'upload':
    st.title("Upload your data for Modelling")
    file = st.file_uploader("Upload your dataset here")
    if file:
        df = pd.read_csv(file , index_col=None)
        df.to_csv("source_data.csv" , index=None)
        st.dataframe(df)
        
# if choice == "Profiling":
#     st.title("Exploratory Data Analysis")
#     if 'df' in locals():
#         profile = ydata_profiling.ProfileReport(df)
#         st_profile_report(profile)
#     else:
#         st.warning("Please upload a dataset first.")

# Modelling Section
if choice == "ML":
    if 'df' in locals():
        st.title("Build Your Model")
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')
            
    else:
        st.warning("Please upload a dataset first.")


if choice  == "Download":
    with open ("best_model.pkl", 'rb') as f:
        st.download_button("Download model" , f, file_name="best_model.pkl")




