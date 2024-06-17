from dotenv import load_dotenv
load_dotenv() 

import streamlit as st
import json
import os
from datetime import datetime, timedelta

import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from transformers import pipeline
from PyPDF2 import PdfReader
import pandas as pd
from pymongo.mongo_client import MongoClient

from streamlit_chat import message
import docx2txt
from PIL import Image
import base64
from io import BytesIO
import time
# import anthropic
# credential_path = "my_venv\client_secret_262550254503-mtga3vurdve1bi3gv161fe6oh0qe62lp.apps.googleusercontent.com.json"
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

from dotenv import load_dotenv
load_dotenv() 

# claude_api_key = os.getenv("CLAUDE_API_KEY")
# client = anthropic.Anthropic(api_key=claude_api_key)


print(genai.configure(api_key=os.getenv("GOOGLE_API_KEY")))

model = genai.GenerativeModel("gemini-1.5-flash-latest")
chat = model.start_chat(history=[])


# MongoDB URI
uri = "mongodb+srv://Usman:DvlNBJrLRyU0gESv@carbonetrix.pllb2my.mongodb.net/?retryWrites=true&w=majority&appName=Carbonetrix"

# MongoDB Management
client = MongoClient(uri)
db = client.get_database("Carbonetrix")
users_collection = db['Carbonetrix']


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []



if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state["messages"].append({"role": "assistant", "content": "Ask Me Anything About The Uploaded Pdfs"})


if 'level' not in st.session_state:
    st.session_state['level'] = 'Beginner'

def add_question_to_db(question, timestamp):
    # company_name = st.sidebar.text_input("PLease enter your company's name for confirmation") # Get the company name from session state

    company_collection = db["Questions"]
    company_collection.insert_one({"question": question, "timestamp": timestamp})


prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

model = ChatGoogleGenerativeAI(model="gemini-pro",
                            temperature=0.3)

prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_file_text(files):
    text = []
    for file in files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text.append(docx2txt.process(file))
        elif file.type == "text/plain":
            text.append(file.getvalue().decode("utf-8"))
        elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(file)
            text.append(df.to_string(index=False))
        else:
            st.error(f"Unsupported file type: {file.type}")
    return text

 
def format_timestamp(timestamp):
    now = datetime.now()
    if timestamp.date() == now.date():
        return "today"
    elif timestamp.date() == (now - timedelta(days=1)).date():
        return "yesterday"
    else:
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
def image_to_base64(image_path):
        img = Image.open(image_path)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return img_str




def main():
    c1, c2, c3 = st.columns([1, 2, 1])
    pth = "Logo.jpg"
    c1.image(pth, width=100)
    c2.title("Chat Your Files")

    chat_container = st.container()
    input_container = st.container()
   
    user_avatar_style = "identicon"
    user_seed = "Angel"


    document1=st.sidebar.file_uploader("Document 1 (question)",accept_multiple_files=True,key="document1")
    document=get_file_text(document1)


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=200)
    context= ", ".join(map(str,document))

    embeddings =SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    text_chunks= text_splitter.split_text(context)

  

    for chuck in text_chunks:
        if embeddings:

            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")


            # embeddings =SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
            c2.markdown('##')
         
            if prompt := st.chat_input():
                timestamp = datetime.now()
                add_question_to_db(prompt, timestamp)

                with input_container:
                    if prompt:
                        for i, msg in enumerate(st.session_state.messages):
                            if msg["role"] == "user":
     
                                st.markdown(f"<div id='{i}' style='text-align: left; color: black; background-color: #dae1e0; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)

                           
                        new_db = FAISS.load_local("faiss_index", embeddings)
                        docs = new_db.similarity_search(prompt)

                        response_text = chain(
                            {"input_documents":docs, "question": prompt}
                            , return_only_outputs=False)
                          
                        timestamp = datetime.now()
                        
                        st.session_state['chat_history'].append(("You", prompt, timestamp))
                        st.session_state['chat_history'].append(("Bot", response_text['output_text'], timestamp))

                        st.session_state["messages"].append({"role": "user", "content": prompt})
                        st.session_state["messages"].append({"role": "assistant", "content": response_text["output_text"]})

                        st.experimental_rerun()

    

    bot_logo_base64 = image_to_base64("Logo.jpg")
    bot_logo_html1 = f"data:image/png;base64,{bot_logo_base64}"

    with chat_container:
        if st.session_state.messages:
            for i, msg in enumerate(st.session_state.messages):
                if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
        
                    message(msg["content"], is_user=True, key=f"user_{i}", avatar_style=user_avatar_style, seed=user_seed,)
                
                elif isinstance(msg, dict) and msg.get("role") == "assistant" and "content" in msg:
                    # Custom HTML for displaying the company logo
                    st.markdown(f"""
                    <div id='{i}' style='text-align: left; color: black; background-color: #f4f4f4; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                        <img src='{bot_logo_html1}' alt='Bot Avatar' style='width: 50px; height: 50px; border-radius: 50%; display: inline-block; vertical-align: middle; margin-right: 10px;' /
                        <span style='vertical-align: middle;'>{msg['content']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                elif not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                    st.error("Invalid message format")

    
    st.sidebar.title("Chat History")
    for entry in st.session_state['chat_history']:
        role, content, timestamp = entry
        if role == "You":
            st.sidebar.write(f"{content} ({format_timestamp(timestamp)})")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)               
          
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask Me Anything About The Uploaded Pdfs"}]
    st.session_state['chat_history'] = []
    st.experimental_rerun()

if __name__ == "__main__":
    main()













