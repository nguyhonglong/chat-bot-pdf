from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

#declare variables
pdf_data_path = "data"
vector_db_path = "vectorstores/dbfaiss"

#
def create_db_from_text():
    raw_text = "lorem amsldm,asld,lá,dlas,ld,asl a,sc  la lshbc  njnasj xj j"


    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    db  = FAISS.from_texts(texts=chunks, embedding = embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_file():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding = embedding_model)
    db.save_local(vector_db_path)

create_db_from_file()