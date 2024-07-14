import streamlit as st
from first_aid_bot import FirstAidBot
from gemini_api import GeminiAPI

# Initialize the First Aid Bot and Gemini API
first_aid_bot = FirstAidBot()
gemini_api = GeminiAPI()

def get_response(user_input):
    # Get first aid advice
    first_aid_advice = first_aid_bot.get_first_aid_advice(user_input)
    
    # Get response from Gemini API
    gemini_response = gemini_api.generate_response(user_input)
    
    return first_aid_advice, gemini_response

# Streamlit UI
st.set_page_config(page_title="First Aid Advice Bot", page_icon="🚑")

st.title("First Aid Advice Bot 🚑")

user_input = st.text_input("Describe the state of emergency or condition:")

if st.button("Get Advice"):
    if user_input:
        first_aid_advice, gemini_response = get_response(user_input)
        st.write(f"### First Aid Advice:\n{first_aid_advice}")
        st.write(f"### Emergency Response:\n{gemini_response}")
    else:
        st.write("Please enter a description of the emergency or condition.")
        st.markdown("---")
st.info("This bot provides first aid advice. For emergencies, always call your local emergency services.")
