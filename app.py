import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
from htmlTemplates import css, bot_template, user_template
import os

def get_pdf_text(pdf_list):
    text = ""
    for pdf_path in pdf_list:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.warning("Please upload the textual PDF file - this is PDF files of image")
        return None
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userInput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    st.session_state.chat_history = []  # Reset chat history after each response

def main():
    load_dotenv()
    st.set_page_config(page_title="AI Medicare", page_icon=".")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""

    # Pre-train the model with the PDF
    sample_pdf_path = os.path.join(os.getcwd(), "Base_conocimiento_Medicare.pdf")
    st.session_state.pdf_files = [sample_pdf_path]

    raw_text = get_pdf_text(st.session_state.pdf_files)
    st.session_state.pdf_text = raw_text
    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)
    st.session_state.conversation = get_conversation_chain(vector_store)
    col1, col2 = st.columns ([1,2])
    with col1:
        st.image('AI_Medicare_logo.png', width=200)
    with col2:
        st.header("AI_Medicare")
        st.text("Tu asistente virtual")
        st.text ("La Inteligencia Artificial al servicio de la salud")
    st.write("<h5><br>Pregunte lo que necesite sobre AI medicare, no importa el idioma, somos multiculturales!:</h5>", unsafe_allow_html=True)
    user_question = st.text_input(label="", placeholder="Dinos quien eres y que haces y podremos ayudarte mejor...")
    if user_question:
        handle_userInput(user_question)

    # Agregar el botón de WhatsApp
    whatsapp_message = "Quiero más info acerca de AI Medicare"
    whatsapp_number = "+5930993513082"
    whatsapp_link = f"https://wa.me/{whatsapp_number}?text={whatsapp_message.replace(' ', '%20')}"
    whatsapp_button = f"""
    <a href="{whatsapp_link}" target="_blank">
        <button style="background-color: #25D366; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
            Mayor info por WhatsApp, contacta aquí
        </button>
    </a>
    """
    st.markdown(whatsapp_button, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
