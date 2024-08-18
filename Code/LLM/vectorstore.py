import time

#from langchain.document_loaders import JSONLoader,TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
import json
def init_DB():
    os.environ['OPENAI_API_KEY'] = "sk-tHGjxn887fj9BsRHidVeeEqAdOYdYkqKrw02102F0EKPOV2Y"
    os.environ['OPENAI_API_BASE'] = "https://api.chatanywhere.com.cn/v1"

    # loader = TextLoader('../Data/cognitive_bias.txt')
    loader = CSVLoader(file_path='./Data/cognitive_bias.csv', encoding='gbk', source_column="biasname")
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db
def get_Knowledge(query,db):
    while True:
        try:
            docs_and_scores = db.similarity_search_with_score(query)
            return docs_and_scores[0][0].page_content
        except:
            time.sleep(10)

    # for i in range(len(docs_and_scores)):
if __name__ == '__main__':
    ans=get_Knowledge('anchoring bias')
    print(ans)
