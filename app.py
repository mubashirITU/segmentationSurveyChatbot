import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
import logging
from utils import load_local_vectordb_using_qdrant,get_conversation_chain

openapi_key = st.secrets["OPENAI_API_KEY"]
qdrant_url = st.secrets["QDRANT_URL"]
qdrant_api_key = st.secrets["QDRANT_API_KEY"]
logging.basicConfig(filename="surveyGPTQdrant.log", format='%(asctime)s %(message)s', filemode='a')
logger = logging.getLogger("surveybot.log")
logger.setLevel(level=logging.DEBUG)
embeddings  = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')


def handel_userinput(user_question,vectorstore):
    # st.session_state.query_history.append(user_question)
    with st.spinner('Wait for the response...'):
        result = st.session_state.conversation({'query': user_question})
        logger.info(f"{st.session_state.chat_history}")
        response = result['result']
        logger.info(f"Result={result}")
    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(f"{response} ")

    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            # print(messages)
            logger.info(f"Answer= {messages}")
            if i % 2 == 0:
                message(messages, is_user=True, key=str(i))
            else:
                message(messages, key=str(i))


def main():
    load_dotenv()
    st.set_page_config(page_title="DMIS chatbot")

    st.header("DMIS Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "memory" not in st.session_state:
        st.session_state.memory = None
    if "check_state" not in st.session_state:
        st.session_state.check_state = False

    with st.sidebar:
        openai_api_key = openapi_key
        st.sidebar.write(""" <div style="text-align: center"> The chatbot will exclusively provide information sourced from our comprehensive survey paper titled 'Semantic Segmentation of Diagnostic Medical Images: Recent Trends and Challenges.' Its primary aim is to efficiently assist users by offering insights and saving their valuable time through targeted and relevant information retrieval.</div>""", unsafe_allow_html=True)
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    vectordb_folder_path = 'dmis_chatbot'

    if st.session_state.vectorstore == None:
        vetorestore = load_local_vectordb_using_qdrant(vectordb_folder_path,embeddings,qdrant_url,qdrant_api_key)
        st.session_state.vectorstore = vetorestore

    if st.session_state.check_state == False:
        st.session_state.conversation = get_conversation_chain(
            st.session_state.vectorstore, openai_api_key)
        st.session_state.check_state = True
    st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        logger.info(f"User Question= {user_question}")
        if user_question:
            handel_userinput(user_question, st.session_state.vectorstore)


if __name__ == '__main__':
    main()