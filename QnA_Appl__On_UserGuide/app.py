
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify, render_template


# # Load environment variables
import os
from dotenv import load_dotenv
load_dotenv()

# Ensure OPENAI_API_KEY is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Pass key to LangChain/OpenAI client automatically
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# # Initialise an embedding function
embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")

#load the index from disk
loaded_vector_store = FAISS.load_local("faiss_index_user_guide", embeddings ,allow_dangerous_deserialization=True)


retriever = loaded_vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not question.strip():
        return jsonify({"error": "Please provide a question"}), 400

    try:
        answer = qa_chain.run(question)
        return jsonify({"question": question, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)