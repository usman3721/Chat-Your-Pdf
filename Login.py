import streamlit as st
# import second_page

from PyPDF2 import PdfReader
import docx2txt
import app as app

c1, c2 = st.columns([4, 7])

def initialize_session_state():
    """Initializes the 'authenticated' key in c1.session_state to False."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
# and password == "admin"
def validate_credentials(password):
    """Validates provided credentials against a defined (dummy) set of credentials.
    Replace this with your actual authentication logic."""
    return password == "12345"

def authenticate_user():
    """Handles user authentication and displays login interface if needed."""
    initialize_session_state()

    if not st.session_state["authenticated"]:
  
        password = st.sidebar.text_input(label="Password", value="", key="passwd", type="password")
        if st.sidebar.button("Access Code"):
            if password:
                authenticated = validate_credentials(password)
                st.session_state["authenticated"] = authenticated

                if not authenticated:
                    st.sidebar.error("Invalid credentials. Please try again.")
    else: 
        st.sidebar.success("Welcome, authenticated user!")
                            
        
        st.markdown("#")
        app.main()      

if __name__ == "__main__":
    authenticate_user()
