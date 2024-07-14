import pandas as pd
import numpy as np
import pickle

import streamlit as st
from streamlit_option_menu import option_menu

copper={"country":[28,25,30,32,38,78,27,77,113,79,26,39,40,84,80,107,89],
        "item type":['W','WI','S','Others','PL','IPL','SLAWR'],
        "product_ref":[1670798778,1668701718,628377,640665,611993,
       1668701376,164141591,1671863738,1332077137,640405,
       1693867550,1665572374, 1282007633, 1668701698,     628117,
       1690738206,     628112,     640400, 1671876026,  164336407,
        164337175, 1668701725, 1665572032,     611728, 1721130331,
       1693867563,     611733, 1690738219, 1722207579,  929423819,
       1665584320, 1665584662, 1665584642]}

with open("C:/Users/DELL XPS/Desktop/python/coppermodel/Lrmodel.pkl","rb") as files:
 mode=pickle.load(files)
with open("C:/Users/DELL XPS/Desktop/python/coppermodel/oheit.pkl","rb") as files:
 ohe1=pickle.load(files)
with open("C:/Users/DELL XPS/Desktop/python/coppermodel/ohes.pkl","rb") as files:
 ohe2=pickle.load(files)
with open("C:/Users/DELL XPS/Desktop/python/coppermodel/Lrscaler.pkl","rb") as files:
 scaler=pickle.load(files)
with open("C:/Users/DELL XPS/Desktop/python/coppermodel/RFCmodel.pkl","rb") as files:
 rfc=pickle.load(files)
with open("C:/Users/DELL XPS/Desktop/python/coppermodel/Classifyscaler.pkl","rb") as files:
 clscaler=pickle.load(files)

st.set_page_config(page_title="Model",
                   page_icon="🔎",
                   layout="wide")

selected = option_menu(None, ["Home", 'Predict Price','Status Check'], 
           icons=['house', 'currency-dollar','stars'], menu_icon="cast", default_index=1,
           orientation="horizontal")
if selected=="Home":
   st.title(":rainbow[Industrial Copper Modeling Prediction]")
   st.image("https://raw.githubusercontent.com/HRLeo19/Coppermodel/main/prediction2.jpg")
if selected=='Predict Price':
    try:
        col1,col2,col3=st.columns(3)
        with col1:
            f5=st.text_input("Enter Quantity in Tons",max_chars=9)
            f6=st.text_input("Enter Thickness of the Material",max_chars=3)
            f4=st.text_input("Enter Width of the Material",max_chars=4)
        with col2:
            f2=st.selectbox("Enter the Country Code",copper["country"])
            f7=st.selectbox("Enter the Product Ref ID",copper["product_ref"])
            f8=st.selectbox("Enter the Item Type",copper["item type"])
        with col3:
            f1=st.text_input("Enter Customer Id",max_chars=8)
            f3=st.text_input("Enter the Application Number",max_chars=3)
            f9=st.selectbox("Select Status",("Won","Lost"))

        on=st.button("Predict the Price")
        if on:
            x=np.array([[f1,f2,f3,f4,f5,f6,f7,f8,f9]])
            item=ohe1.transform(x[:,[7]]).toarray()
            status=ohe2.transform(x[:,[8]]).toarray()
            x=np.concatenate((x[:,[0,1,2,3,4,5,6]],item,status),axis=1)
            x1=scaler.transform(x)
            predict=mode.predict(x1)[0]
            st.success(f"### The Predicted Price is 🌟{predict} 💸💸💸")
    except:
       st.write("#### **Please Enter Details Properly,All Boxes to be filled only with Numbers.**")

if selected=="Status Check":
    try:
        co1,co2,co3=st.columns(3)
        with co1:
            fc5=st.text_input("Enter Quantity in Tons",max_chars=9)
            fc6=st.text_input("Enter Thickness of the Material",max_chars=3)
            fc4=st.text_input("Enter Width of the Material",max_chars=4)
        with co2:
            fc2=st.selectbox("Enter the Country Code",copper["country"])
            fc7=st.selectbox("Enter the Product Ref ID",copper["product_ref"])
            fc8=st.selectbox("Enter the Item Type",copper["item type"])
        with co3:
            fc1=st.text_input("Enter Customer Id",max_chars=8)
            fc3=st.text_input("Enter the Application Number",max_chars=3)
            fc9=st.text_input("Enter the Price",max_chars=3)

        onn=st.button("Predict the Status")
        if onn:
            xc=np.array([[fc1,fc2,fc3,fc4,fc5,fc6,fc7,fc8,fc9]])
            itemc=ohe1.transform(xc[:,[7]]).toarray()
            xc=np.concatenate((xc[:,[0,1,2,3,4,5,6,8]],itemc),axis=1)
            x1c=clscaler.transform(xc)
            predict_cc=rfc.predict(x1c)[0]
            if predict_cc=="Won":
                st.success(f"### **The Status is 🌟:green[{predict_cc}] 📢📢📢.More chance that lead will become regular customer.**")
            else:
                st.write(f"### **Lead is :red[**{predict_cc}**] 😱.Less Likely become regular Customer**")
    except:
       st.write("#### **Please Enter Details Properly,All Boxes to be filled only with Numbers.**")
