from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS


#config
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/dbfaiss"


#load LLM
def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_tokens = 1024,
        temperature = 0.01
    )

    return llm

def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "question"])
    return prompt

def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}),
        chain_type_kwargs = {"prompt": prompt}
    )
    return llm_chain

def read_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db 

db = read_vector_db()
llm = load_llm(model_file)

template = template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, đừng cố tạo ra câu trả lời
{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

question = "Một tín chỉ được quy định tối thiểu bằng bao nhiêu giờ?"
response = llm_chain.invoke({"query":question})
print(response)