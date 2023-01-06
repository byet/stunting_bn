import streamlit as st
import pandas as pd
import pyAgrum as gum
import matplotlib.pyplot as plt


if 'bn' not in st.session_state:
    st.session_state['bn'] = gum.BayesNet("bn2")
    st.session_state.bn.loadNET("bn2.net")
    target = "Stunting"
    st.session_state['target'] = target
    st.session_state['target_states'] = st.session_state.bn.variable(target).labels()
    vars = st.session_state.bn.names()
    vars.remove(st.session_state.target)
    st.session_state['vars'] = list(vars)
    st.session_state['ie'] = gum.LazyPropagation(st.session_state.bn)
    states = dict()
    for var in vars:
        states[var] = st.session_state.bn.variable(var).labels()
    st.session_state['states'] = states
    #st.session_state['post'] = gnb.getInference(st.session_state['bn'], evs={})
    st.session_state.ie.makeInference()
   #st.session_state['post'] = gnb.getPosterior(st.session_state['bn'], evs={}, target=st.session_state.target)
    st.session_state['post'] = st.session_state.ie.posterior(st.session_state.target).toarray()
    #st.session_state['ie'] = gum.LazyPropagation(st.session_state['bn'])



def callback():
    evs = {}
    for var in st.session_state.vars:
        if st.session_state[var]  != 'unknown':
            evs[var] = st.session_state[var] 
        #st.session_state['post'] = gnb.getInference(st.session_state['bn'], evs=evs)
        #st.session_state['post'] = gnb.getPosterior(st.session_state['bn'], evs=evs, target=st.session_state.target)
        st.session_state.ie.eraseAllEvidence()
        st.session_state.ie.setEvidence(evs)
        st.session_state.ie.makeInference()
        st.session_state['post'] =   st.session_state.ie.posterior(st.session_state.target).toarray()

st.header("Stunting Bayesian Network", anchor=None)

col1, col2 = st.columns(2)


with col1:
    st.subheader("Predictors", anchor=None)
    for var in st.session_state.vars:
        var_states = st.session_state.states[var] + ('unknown',)
        st.radio(var,var_states,key=var, on_change=callback, horizontal=True, index=len(var_states) - 1)


with col2:
    st.subheader("Stunting", anchor=None)
    #st.markdown(st.session_state.post,unsafe_allow_html=True)
    #st.bar_chart(st.session_state.post)
    
    states = st.session_state.target_states   
    probs = st.session_state.post
    fig = plt.figure(figsize = (4, 6))

    # creating the bar plot
    plt.style.use('dark_background')
    plt.bar(states, probs, color ='red', width = 0.4)
    plt.xlabel("Stunting")
    plt.ylabel("Probability")
    plt.ylim(0,1)
    st.pyplot(plt, transparent=True)
    st.markdown(f"**Pr({st.session_state.target} = yes) = {st.session_state.post[1]:.2f}**")

  


# st.session_state.states
