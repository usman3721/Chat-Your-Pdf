# Libraries
import streamlit as st
from PIL import Image





st.set_page_config(page_title='Document Comparer App', page_icon=':bar_chart:', layout='wide')

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.image(Image.open('download.png'))


c2.title('Document Comparer App')

# .stSidebarNavSeparator {
#   pointer-events: none;
# }

# Introduction
st.subheader('Introduction')
st.write(
    """
Welcome to "Chat Your PDF"!

Unlock the power of conversation with your PDF documents like never before. Say goodbye to tedious scrolling and searching through pages of text. With "Chat Your PDF," you can effortlessly interact with your PDF files through simple, natural language.
"""
)

st.subheader('Key Features')


st.write("""
Natural Language Interaction: Engage with your PDF documents using natural language commands and queries, making it easy to navigate, search, and extract information.

PDF Parsing: Seamlessly parse PDF files to extract text, images, and other relevant content, enabling efficient interaction with the document's contents.

Intelligent Search: Utilize advanced search capabilities to quickly find specific information within your PDF files, enhancing productivity and saving time.

Multi-format Support: Support for various file formats, including PDF, DOCX, and TXT, allowing users to upload and interact with documents in different formats.

Conversation History: Maintain a history of your interactions with each PDF document, enabling you to track your progress and revisit previous queries and commands.

Annotation and Highlighting: Annotate and highlight sections of your PDF documents directly within the chat interface, facilitating collaboration and knowledge sharing.

Personalization: Customize the chat interface and settings to suit your preferences, including font size, theme, and language options.

Responsive Design: Ensure compatibility across devices with a responsive web design that adapts to various screen sizes and orientations, providing a seamless user experience on desktops, tablets, and smartphones.

Secure Data Handling: Implement robust security measures to protect user data and ensure confidentiality when interacting with sensitive documents.

Integration with Cloud Storage: Integrate with popular cloud storage services such as Google Drive, Dropbox, and OneDrive, allowing users to access and interact with their PDF files directly from their cloud accounts.

Feedback and Support: Provide users with a feedback mechanism and access to customer support resources to address inquiries, resolve issues, and gather suggestions for future enhancements.

Accessibility Features: Incorporate accessibility features such as screen reader compatibility and keyboard navigation, ensuring inclusivity and usability for all users, including those with disabilities.
    """
)

# Methodology
st.subheader('Methodology')
st.write(
    """
 Parse uploaded PDF files to extract text and metadata.

Utilize natural language processing (NLP) algorithms to interpret user queries and commands.

Implement search and navigation algorithms to locate relevant content within the documents.

Enable interactive chat-based communication for users to interact with PDF content seamlessly 
# [**GitHub Repository**](https://github.com/alitaslimi/cross-chain-monitoring).
    """

)

# Divider
st.divider()



# st.page_link("http://localhost:8501/Login", label="Login", icon="2️⃣")
# st.sidebar.page_link("http://localhost:8501/Home", label="Manage users")








    



