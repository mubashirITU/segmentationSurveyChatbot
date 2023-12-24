import os
import logging
import csv
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS,Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from typing import List,Tuple
from qdrant_client import QdrantClient
import openai

logging.basicConfig(filename="surveyGPTQdrant_utils.log", format='%(asctime)s %(message)s', filemode='a')

logger = logging.getLogger("survey_utils.log")
logger.setLevel(level=logging.DEBUG)




def _format_chat_history(chat_history: List[Tuple]) -> str:
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer



def get_conversation_chain(vectorstore, openai_key):
    num_chunks=30

    prompt_template="""
    As a research expert in semantic segmentation of diagnostic medical images, you have access to the following paragraphs from a survey paper on this topic. You need to answer the query by analyzing the paragraphs and extracting relevant information.
    If you doon't able to find any relevant information, then return: "How can I help you? Please ask question related to 'Semantic Segmentation of Diagnostic Medical Images: Recent Trends and Challenges'".

    ## Paragraphs:
    {context}

    ## Query:
    {question}

    ## Answer:
    •  Format your answer as bullet points or sentences, depending on the query.

    •  If possible, cite facts and evidence from the paragraphs to back up your answer.
    """
    mprompt_url = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"], validate_template=False)
    chain_type_kwargs = {"prompt": mprompt_url}

    chat_llm = ChatOpenAI(model = "gpt-3.5-turbo-16k", openai_api_key = openai_key , temperature=0.3)
    multiqa_retriever = MultiQueryRetriever.from_llm(llm=chat_llm, 
                                         retriever=vectorstore.as_retriever(search_kwargs={"k": num_chunks}))
    qa = RetrievalQA.from_chain_type(llm=chat_llm, 
                                     chain_type="stuff", 
                                    #  retriever=local_db.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks}), 
                                     retriever = multiqa_retriever,
                                     chain_type_kwargs=chain_type_kwargs, 
                                     return_source_documents=True)
    return qa
 

def create_new_vectorstore_qdrant( doc_list, embed_fn, COLLECTION_NAME,qdrant_url, qdrant_api_key ):
    try:
        qdrant = Qdrant.from_documents(
            documents = doc_list,
            embedding = embed_fn,
            url=qdrant_url,
            prefer_grpc=True,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Successfully created the vectordb")
        return qdrant
    except Exception as ex:
        logger.exception("Vectordb Failed:"+str(ex))
        # return JSONResponse({"Error": str(ex)})

def load_local_vectordb_using_qdrant( vectordb_folder_path, embed_fn, qdrant_url, qdrant_api_key):
    try:
        qdrant_client = QdrantClient(
            url=qdrant_url, 
            prefer_grpc=True,
            api_key=qdrant_api_key,
        )
        logger.info("Qdrant client loaded Successfully")
        qdrant_store= Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
        logger.info("Successfully loaded vectordb")
        return qdrant_store   
    except Exception as e:
        logger.critical(f"error while loading vectordb:'{str(e)}'")
        raise Exception(f"error while loading vectordb:'{str(e)}'")
    
