import streamlit as st


with open('./resources/evidence.json', 'r') as file:
    evidence = file.read()

st.json(evidence)