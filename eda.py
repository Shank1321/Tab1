import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn import metrics as mat
#pip install pickle-mixin
import pickle

st.set_page_config(page_title="Fruits Data Analysis",page_icon=":green_salad:",layout="wide")

st.title(":apple: Fruits Data Analysis :lemon:")

df=pd.read_csv('fruits.csv')

st.header('Fruits Dataset')
st.table(df.head())

st.header('Fruits Dataset Summary')
st.table(df.describe())

st.header('Data Visualization')
fig1=px.scatter(df,x="width",y="height",color="color_score")
st.plotly_chart(fig1,use_container_width=True)

st.subheader("Mass")
fig2=px.histogram(df,x="mass",nbins=20)
st.plotly_chart(fig2,use_container_width=True)

st.subheader("Boxplot with Height and Width ")
fig3=px.box(df,x="height",y="width",color="fruit_name")
st.plotly_chart(fig3,use_container_width=True)

#Converting columns
le=LabelEncoder()
cat_col=['fruit_name','fruit_subtype','mass']
for col in cat_col:
    df[col]=le.fit_transform(df[col])

st.header('Updated Fruits Dataset')
st.table(df.head())

x=df.drop(columns=['color_score'],axis=1)
y=df[['color_score']]

c1,c2=st.columns(2)
c1.subheader('Feature Set')
c1.table(x.head())

c2.subheader('Target')
c2.table(y.head())

# Diving to training and test set
xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.3,random_state=10,shuffle=True)

c3,c4,c5,c6=st.columns(4)

c3.subheader('Train Feature Set')
c3.table(xtrain.head())

c4.subheader('Train Labels Set')
c4.table(ytrain.head())

c5.subheader('Test Feature Set')
c5.table(xtest.head())

c6.subheader('Test Label Set')
c6.table(ytest.head())

# Regression models
gbr=GradientBoostingRegressor(max_depth=2,n_estimators=5,learning_rate=1.0)
gbr.fit(xtrain,ytrain)
ypred1=gbr.predict(xtest)
r1=round(mat.r2_score(ytest,ypred1))
m1=pickle.dump(gbr,open('gbr1.pkl','wb'))

adr=AdaBoostRegressor()
adr.fit(xtrain,ytrain)
ypred2=adr.predict(xtest)
r2=round(mat.r2_score(ytest,ypred2))
m2=pickle.dump(adr,open('adr1.pkl','wb'))

# Comparison of models
c7,c8=st.columns(2)
c7.subheader('Gradient Boosting Regressor model')
c7.subheader(r1)
c8.subheader('Ada Boost Regressor model')
c8.subheader(r2)

