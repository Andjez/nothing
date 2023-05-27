import streamlit as st

# Set a title for the web app
st.title("Simple Streamlit App")

# Add a text input box
user_input = st.text_input("Enter your name")

# Display the input value
st.write("Hello, " + user_input + "! Welcome to Streamlit.")
