import streamlit as st

# Basic app configuration
st.set_page_config(page_title="Test App")

# Simple header
st.title("Simple Test App")
st.write("Hello! This is a test app.")

# Add a simple interactive element
if st.button("Click me!"):
    st.write("Button clicked!")
