import streamlit as st

st.title(body="Departments")

# Create Tabs
tabTitles=["HR Support","IT Support","Transport Support"]
tabs=st.tabs(tabs=tabTitles)

with tabs[0]:
    st.header(body="HR Support tickets")
    for ticket in st.session_state['HR_tickets']:
        st.write(str(st.session_state['HR_tickets'].index(ticket)+1)+" : "+ticket)
    

with tabs[1]:
    st.header(body="IT Support tickets")
    for ticket in st.session_state['IT_tickets']:
        st.write(str(st.session_state['IT_tickets'].index(ticket)+1)+" : "+ticket)

with tabs[2]:
    st.header(body="Transportation Support tickets")
    for ticket in st.session_state['Transport_tickets']:
        st.write(str(st.session_state['Transport_tickets'].index(ticket)+1)+" : "+ticket)