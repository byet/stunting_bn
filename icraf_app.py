import streamlit as st
import pandas as pd
import pyAgrum as gum
import pyAgrum.lib.notebook as gnb


if 'bn' not in st.session_state:
    st.session_state['bn'] = gum.BayesNet("bn2")
    st.session_state.bn.loadNET("bn2.net")
    st.session_state['target'] = "Stunting"
    vars = st.session_state.bn.names()
    vars.remove(st.session_state.target)
    st.session_state['vars'] = list(vars)

    states = dict()
    for var in vars:
        states[var] = st.session_state.bn.variable(var).labels()
    st.session_state['states'] = states
    #st.session_state['post'] = gnb.getInference(st.session_state['bn'], evs={})
    st.session_state['post'] = gnb.getPosterior(st.session_state['bn'], evs={}, target=st.session_state.target)
    #st.session_state['ie'] = gum.LazyPropagation(st.session_state['bn'])



def callback():
    evs = {}
    for var in st.session_state.vars:
        if st.session_state[var]  != 'unknown':
            evs[var] = st.session_state[var] 
        #st.session_state['post'] = gnb.getInference(st.session_state['bn'], evs=evs)
        st.session_state['post'] = gnb.getPosterior(st.session_state['bn'], evs=evs, target=st.session_state.target)
    pass

st.header("Stunting Bayesian Network", anchor=None)

col1, col2 = st.columns(2)


with col1:
    st.subheader("Predictors", anchor=None)
    for var in st.session_state.vars:
        var_states = st.session_state.states[var] + ('unknown',)
        st.radio(var,var_states,key=var, on_change=callback, horizontal=True, index=len(var_states) - 1)


with col2:
    st.subheader("Stunting", anchor=None)
    st.markdown(st.session_state.post,unsafe_allow_html=True)

# st.session_state.states
