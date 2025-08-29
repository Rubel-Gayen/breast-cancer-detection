import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Breast Cancer detection system')

df=pickle.load(open('cancer.pkl','rb'))
pipe=pickle.load(open('pipe.pkl','rb'))
data=pd.DataFrame(df)
op=st.selectbox("Enter the Clump Thickness", data['Clump Thickness'].unique())
op1=st.selectbox("Enter the Uniformity of cell size", data['Uniformity of Cell Size'].unique())
op2=st.selectbox("Enter the Uniformity of cell shape", data['Uniformity of Cell Shape'].unique())
op3=st.selectbox("Enter the Marginal Adhesion", data['Marginal Adhesion'].unique())
op4=st.selectbox("Enter the Single Epithelial Cell size", data['Single Epithelial Cell Size'].unique())
op5=st.selectbox("Enter the bare nuclei", data['Bare Nuclei'].unique())
op6=st.selectbox("Enter the Bland Chromatin", data['Bland Chromatin'].unique())
op7=st.selectbox("Enter the Normal Nucleoli", data['Normal Nucleoli'].unique())
op8=st.selectbox("Enter the mitosis", data['Mitoses'].unique())

if(st.button("Predict")):
    query=np.array([op,op1,op2,op3,op4,op5,op6,op7,op8])
    query=query.reshape(1,9)
    s=((pipe.predict(query)))

    if(s[0] == 2):
        st.title("Benign(non-cancerous)")
    else:
        st.title("Malignant(cancerous)")