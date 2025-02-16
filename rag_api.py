import os
import boto3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_aws.llms.bedrock import BedrockLLM
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Apply CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Directory containing PDFs
pdf_folder = "pdf"

documents = []
for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        documents.extend(loader.load())

# Splitting the data into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
docs = text_splitter.split_documents(documents=documents)

# Loading the embedding model from HuggingFace
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

# 🔹 Try to Load FAISS, if it doesn’t exist, create it
faiss_index_path = "faiss_index"
if not os.path.exists(faiss_index_path):  
    print("FAISS index not found! Creating a new one...")  
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(faiss_index_path)
else:
    print("FAISS index found. Loading from disk...")  
    vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Creating a retriever on top of the database
retriever = vectorstore.as_retriever()

# 🔹 Initialize AWS Bedrock Client (Mistral)
bedrock_llm = BedrockLLM(
    model_id="mistral.mistral-large-2402-v1:0",
    client=boto3.client(
        service_name="bedrock-runtime",
        region_name="us-east-1",  # Change to your actual AWS region
        config=boto3.session.Config(proxies={})  # Disables proxy usage
    )
)


# 🔹 Use RetrievalQA with Bedrock as LLM
qa = RetrievalQA.from_chain_type(llm=bedrock_llm, chain_type="stuff", retriever=retriever)

# Define request model
class QueryRequest(BaseModel):
    query: str

# API endpoint for query processing
@app.post("/query")
async def query_pdf(request: QueryRequest):
    if request.query.lower() == "exit":
        return {"response": "Session terminated."}
    
    try:
        result = qa.run(request.query)
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run FastAPI with Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
