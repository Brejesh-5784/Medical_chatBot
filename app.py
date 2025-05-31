from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone  # Updated import
import pinecone
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate, ChatPromptTemplate

app = Flask(__name__)

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define the prompt template
prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:
"""

# System prompt for the new approach
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create prompt templates
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs = {"prompt": PROMPT}

# Initialize LLM
try:
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-70b-8192",
        temperature=0.4,
        timeout=30,
        max_retries=3,
    )
    # Test the connection
    response = llm.invoke("Hello, how are you?")
    print("LLM Connection successful:", response.content[:100] + "...")
except Exception as e:
    print(f"Error initializing LLM: {e}")

# Method 1: Using RetrievalQA (your current approach)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Method 2: Using create_retrieval_chain (alternative approach)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_chain = create_stuff_documents_chain(llm, chat_prompt)
rag_chain = create_retrieval_chain(retriever, question_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg
    print(f"User input: {input_text}")
    
    try:
        # Using Method 1 (RetrievalQA)
        result = qa({"query": input_text})
        response_text = result["result"]
        print("Response:", response_text)
        
        # Alternative: Using Method 2 (create_retrieval_chain)
        # result = rag_chain.invoke({"input": input_text})
        # response_text = result["answer"]
        
        return str(response_text)
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return "Sorry, I encountered an error. Please try again."

if __name__ == '__main__':
    import socket
    
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    port = find_free_port()
    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)