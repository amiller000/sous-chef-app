import pandas as pd
from dotenv import load_dotenv
import os
import openai

from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

recipes_df = pd.read_csv("/Users/alexmiller/Documents/data/recipes/dataset/full_dataset.csv")
recipes = recipes_df.iloc[0:50000]
recipes['directions'] = recipes['directions'].apply(lambda x: x.replace("[", "").replace('"', "").replace("]", "").replace(".,", "."))
recipes['ingredients'] = recipes['ingredients'].apply(lambda x: x.replace("[", "").replace('"', "").replace("]", "").replace(".,", "."))
recipes['text'] = recipes['title'] + ' ' + recipes['directions'] + ' ' + recipes['ingredients']

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)
text_input = recipes.text.apply(lambda r: text_splitter.split_text(r)[0]).to_list()
metadata_inputs = (recipes
                  .drop(['text', 'Unnamed: 0', 'NER'], axis=1)
                  .to_dict(orient='records'))

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# instantiate vector store object
vector_store = FAISS.from_texts(
  embedding=embeddings, 
  texts=text_input,
  metadatas=metadata_inputs
  )

vector_store.save_local(folder_path="/Users/alexmiller/Documents/data/recipes/faiss/recipe_embeddings")