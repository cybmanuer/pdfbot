import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

# Configure API Key
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Helper Functions
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text


#step 1
# function that reads the files selected by the users and check error if the file can'tbe read or photocopied pdf 
# onceit is read the text  and return it to this function get_text_chunks(text)
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text is extracted
                    text += page_text
                # else:
                #     st.warning(f"No text found on page of {pdf.name}. It may be a scanned document.")
        except Exception as e:
            st.error(f"Error reading file ->  {pdf.name}: {str(e)}. The file may be encrypted or corrupted.")
    return text



# STEP 2 
# This function converts the text into small chunks of 10000 words or char 
# then it returns the chunks to the function get_vector_store(text_chunks)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


#STEP 3
# This function converts the chuns text to vectors and store it localy
# this stored chuncs are like database it used in extarcting the requried information by AI
def get_vector_store(text_chunks):
    if not text_chunks:  # Check if there are no chunks to process
        st.error("No text chunks available for vector store creation. Please ensure the PDFs contain readable text.")
        return exit # Exit the function early if no chunks are available
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# STEP  5   
#this function is usefull for communicating or intarcting with GEMINI AI 
# this function hold the prompt the requried to fetch the correct answer 
# this function sends PROMPT,DATA and QUESTION asked by user to AI to fetch the answer.
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\nA
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# STEP 4
# this function is responsible for geting user input  and handles user queries by leveraging vector embeddings and similarity search
# 
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  #specifying a model for generating embeddings. Embeddings are dense vector representations of text that capture semantic meaning
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  #The function loads a pre-built FAISS index from local storage using the specified embeddings model. FAISS (Facebook AI Similarity Search) is an efficient library for similarity search and clustering of dense vectors
    docs = new_db.similarity_search(user_question)  #This line performs a similarity search using the user’s question as the query. The similarity_search method compares the vector representation of the question against the vectors in the FAISS index to find documents that are semantically similar.
    
    # This line passes both the retrieved documents (docs) and the user's question to the conversational chain to generate a response. 
    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write(f"<div style='font-size: 18px; color: white; font-family: Arial;'><b>Reply &#x1F449 :</b> {response['output_text']}</div>", unsafe_allow_html=True)


# Main Function
def main():
    # Page Configurations
    st.set_page_config(page_title="PDF IntelliBot", page_icon="🤖", layout="wide")

    # Custom CSS Styles
    st.markdown(
        """
        <style>
        *{
            
        }
        .header-title {
            font-family: 'Georgia', serif;
            color: radial-gradient(circle, rgba(205,78,194,1) 0%, rgba(136,62,201,1) 100%);;
            text-align: center;
            font-size: 36px;
            margin-bottom: 20px;
            # background: rgb(205,78,194);
            background: black;
        }
        .sub-title {
            font-family: 'Verdana', sans-serif;
            color: #5D6D7E;
            font-size: 18px;
            margin-bottom: 30px;
            
        }
        .sidebar {
            background-color: #F8F9F9;
        }
        .uploaded-file-section {
            font-family: 'Tahoma', sans-serif;
            # color: #34495E;
            font-size: 16px;

        }
       
        .text-input {
            font-family: 'Courier New', monospace;
            color: #283747;
            font-size: 18px;
            color:#ECDFCC;
        }
      
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header Section
    st.subheader(" Your Document Answer Engine")
    st.markdown(
        "<div class='sub-title'>Upload your PDF files, ask questions, and get intelligent responses powered by AI.</div>",
        unsafe_allow_html=True,
    )

    # User Input for Questions
    user_question = st.text_input("Ask a Question", placeholder="Type your question here...", key="user_question", help="Enter your query about the uploaded PDF files and Enter .")
    if user_question:
        user_input(user_question)

    # Sidebar for File Upload
    with st.sidebar:
        st.markdown("<div class='header-title'>PDF IntelliBot </div><br><br>", unsafe_allow_html=True)

        st.markdown("<div class='uploaded-file-section'><b>Upload PDFs</b></div>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader(
            "Upload your PDF Files (Multiple Allowed)",
            accept_multiple_files=True,
            key="pdf_upload",
        )
        
        st.markdown("<div><br></div>",unsafe_allow_html=True)
        process_button = st.button("Upload & Process", help="Click to process the uploaded PDFs.")
        st.markdown("</p>",unsafe_allow_html=True)



        if process_button:
            if not pdf_docs:  # Check if no files are selected
                st.error("No files selected. Please upload at least one PDF file.", icon="🚨")
            else:
                with st.spinner("Processing your PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete ✅", icon="✔️")

        

# Run Application
if __name__ == "__main__":
    main()
