import pandas as pd
from dotenv import load_dotenv
import os
import openai

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import OpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain import LLMChain

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

folder_path = "/Users/alexmiller/Documents/data/recipes/faiss/recipe_embeddings"

# define model to respond to prompt
def load_qa_chain(model_name='gpt-3.5-turbo', temperature=0.1):
    # define system-level instructions
    system_message_template = """<|im_start|> 
                                You are a helpful cooking assistant named Sous, you are good at recommending recipes based on the input query from the user and recipe context. 
                                If the query doesn't form a complete question, just say I don't know. 
                                If there is a good answer from the context, provide the best recipe based on the answer. 
                                If you don't know the answer, just say I don't know. 
                                If there is no enough information to determine the answer, just say I don't know. 
                                If the context is irrelevant to the question, then answer the question with the best of your ability. 
                                Always answer the user with "Yes Chef!" and then provide the recipe recommendation. 
                                <|im_end|>"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_template)

    # define human-driven instructions
    human_message_template = """<|im_start|> Here is the question {question}. Here is the context: {context}? <|im_end|>"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_message_template)

    # combine instructions into a single prompt
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    llm = OpenAI(model_name=model_name, temperature=temperature)

   # combine prompt and model into a unit of work (chain)
    qa_chain = LLMChain(
        llm = llm,
        prompt = chat_prompt
        )
    
    return qa_chain

def load_retriever(model='text-embedding-ada-002', folder_path=folder_path, n_documents=5):

    # open vector store to access embeddings
    embeddings = OpenAIEmbeddings(model=model)
    vector_store = FAISS.load_local(embeddings=embeddings, folder_path=folder_path)

    # configure document retrieval 
    retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism
    return retriever

def retrieve_documents(query, retriever):

    # get relevant documents
    docs = retriever.get_relevant_documents(query)
    return docs

def get_answer(docs, query, qa_chain):

    for doc in docs:
        # get document text
        text = doc.page_content

        # generate a response
        output = qa_chain.generate([{'context': text, 'question': query}])
        
        # get answer from results
        generation = output.generations[0][0]
        answer = generation.text

        # display answer
        if answer is not None:
            return answer
        
    return "No answer returned"

# qa_chain = load_qa_chain()
# retriever = load_retriever()

# query = "I am wanting to cook Greek dish that includes chicken, vegetables, and rice. Recommend a recipe that is quick, healthy, and delicious and can be baked in the oven."

# docs = retrieve_documents(query=query, retriever=retriever)
# answer = get_answer(docs, query, qa_chain)

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/recommend_recipe", response_model=AnswerResponse)
def recommend_recipe(query_request: QueryRequest):
    query = query_request.query

    # Your existing code
    qa_chain = load_qa_chain()
    retriever = load_retriever()
    docs = retrieve_documents(query=query, retriever=retriever)
    answer = get_answer(docs, query, qa_chain)

    return AnswerResponse(answer=answer)

import uvicorn

if __name__ == "__main__":
    # uvicorn.run(app, host="127.0.0.1", port=8000)
    uvicorn.run(app, host="0.0.0.0", port=8080)
