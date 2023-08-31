from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Hardcoded API key
openai_api_key = "Your API Key"


def load_document(filename):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(filename)
        documents = loader.load()
    elif filename.endswith(".txt"):
        loader = TextLoader(filename)
        documents = loader.load()
    else:
        raise ValueError("Invalid file type")

    text_splitter = CharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    return docs


def query_pdf(query, retriever):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),
                                     chain_type="stuff", retriever=retriever)
    result = qa.run(query)
    print(result)


def main():
    filename = input("Enter the name of the document (.pdf or .txt):\n")
    docs = load_document(filename)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_constitution")
    persisted_vectorstore = FAISS.load_local("faiss_index_constitution", embeddings)

    query = input("Type in your query (type 'exit' to quit):\n")
    while query != "exit":
        query_pdf(query, persisted_vectorstore.as_retriever())
        query = input("Type in your query (type 'exit' to quit):\n")


if __name__ == "__main__":
    main()
