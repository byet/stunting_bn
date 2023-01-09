import streamlit as st
import pyAgrum.lib.notebook as gnb


st.markdown(gnb.getInference(st.session_state.bn), unsafe_allow_html=True)