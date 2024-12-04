import streamlit as st
import pickle

st.set_page_config(page_title="Color Score prediction",
page_icon=":green_salad:",layout="wide")

st.title(":large_yellow_circle: Color Score prediction :large_orange_circle:")
model1=pickle.load(open('gbr1.pkl','rb'))
model2=pickle.load(open('adr1.pkl','rb'))
c1,c2=st.columns(2)
n1=c1.number_input("fruit_label?")
n2=int(c2.number_input("fruit_name?"))
n3=int(c1.number_input("fruit_subtype?"))
n4=int(c2.number_input("mass?"))
n5=c1.number_input("width")
n6=c2.number_input("height?")
new_feature=[[n1,n2,n3,n4,n5,n6]]
c3,c4=st.columns(2)
if c3.button("GBR Model prediction"):
   t1=model1.predict(new_feature)
   c3.subheader(t1)

if c4.button("ADR Model prediction"):
   t2=model2.predict(new_feature)
   c4.subheader(t2)