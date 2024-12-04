import streamlit as st
pg=st.navigation([st.Page("eda.py",title="Fruits Data Analysis"),
st.Page("model.py",title="Color Score Prediction")])
pg.run()