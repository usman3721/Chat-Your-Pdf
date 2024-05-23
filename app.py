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
from transformers import pipeline
from PyPDF2 import PdfReader
import docx2txt
import time

print(genai.configure(api_key=os.getenv("GOOGLE_API_KEY")))

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

time.sleep(0.1)


if 'level' not in st.session_state:
    st.session_state['level'] = 'Beginner'

# Add initial assistant message if chat history is empty
if not st.session_state["messages"]:
    st.session_state["messages"].append({"role": "assistant", "content": "Ask Me Anything About The Uploaded Pdfs"})










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
        if file.type == "application/pdf":  # Handle PDF files
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # Handle DOCX files
            text.append(docx2txt.process(file))
        elif file.type == "text/plain":  # Handle text files
            with file.open(encoding="utf-8") as f:  # Adjust encoding if needed
                text.append(f.read())
        else:  # Handle unsupported file types
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


# document1=st.sidebar.file_uploader("Document 1 (question)",accept_multiple_files=True,key="document1")
# document=get_file_text(document1)


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=200)
# context= ", ".join(map(str,document))



# text_chunks= text_splitter.split_text(context)


 


def main():
    c1, c2, c3 = st.columns([1, 2, 1])
    pth = "Logo.jpg"
    c1.image(pth, width=130)
    c2.title("Chat Your Pdfs")



    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        # for msg in st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):

            if msg["role"] == "user":
                # st.markdown(f"<div style='text-align: left; color: black; bac kground-color: #90EE90; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
                # st.markdown(f"<div style='text-align: right; color: black; background-color: #d3d3d3; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div id='{i}' style='text-align: left; color: black; background-color: #dae1e0; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)

            else:
                # st.markdown(f"<div style='text-align: left; color: black; background-color: #90EE90; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
                # st.markdown(f"<div style='text-align: left; color: white; background-color: #1a73e8; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div id='{i}' style='text-align: left; color: black; background-color: #f4f4f4; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)

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

                with input_container:
                    if prompt:
                        with chat_container:
                            for i, msg in enumerate(st.session_state.messages):
                                if msg["role"] == "user":
                                        st.markdown(f"<div id='{i}' style='text-align: left; color: black; background-color: #dae1e0; padding: 10px; border-radius: 10px; margin: 10px 0;'>{msg['content']}</div>", unsafe_allow_html=True)




                        new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
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















