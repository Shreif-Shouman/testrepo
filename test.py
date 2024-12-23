import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import HuggingFaceInstructEmbeddings


# Function to save uploaded files
def save_uploaded_file(uploaded_file, directory):
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Function to load FAISS index
def load_faiss_index(folder_path, index_name):
    embeddings = HuggingFaceInstructEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    faiss_index = FAISS.load_local(folder_path, embeddings, index_name, allow_dangerous_deserialization=True)
    return faiss_index


# Function to get conversation chain
def get_conversation_chain(vectorstore):
    template = """
    Use the following information to answer the user's question in **ONLY ONE SENTENCE**.

    Be concise and brief. Do not provide detailed explanations or additional information.

    Context: {context}
    Question: {question}
    Your one-sentence answer:
    """

    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    
    api_key = os.environ['HUGGINGFACEHUB_API_TOKEN'] = st.secrets['HUGGINGFACEHUB_API_TOKEN']
    llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", huggingfacehub_api_token=api_key)

    retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 3})
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=retriever, base_compressor=compressor
    )

    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": prompt, "verbose": True},
        return_source_documents=True
    )

    return conversation_chain


# Function to handle user input and return answer
def handle_userinput(user_question):
    # Check if the conversation chain is initialized
    if st.session_state.conversation is None:
        st.warning("Please upload and load the FAISS index before asking questions.")
        return
    
    # Get the response from the conversation chain
    response = st.session_state.conversation({'query': user_question})  # Use 'question' key
    
    # Extract the answer and sources
    answer = response.get('result', "No answer available.")
    sources = response.get('source_documents', [])

    # Display the question and the answer
    st.markdown(f"**Question:** {user_question}")
    st.markdown(f"**Answer:** {answer}")
    
    # Display sources, if available
    if sources and answer.strip():
        st.markdown("**Cited Sources:**")
        for source in sources:
            url = source.metadata.get('url', 'URL not available')
            page_content = source.page_content[:1400]  # Limit the length of the text to 1400 characters
            st.markdown(f"<p><strong>Source:</strong> <a href='{url}' target='_blank'>{url}</a></p>", unsafe_allow_html=True)
            st.markdown(f"<p><strong>Text Used:</strong> {page_content}...</p>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="FAISS PDF QA", layout="centered")

    custom_html = """
    <div style="text-align: center;">
        <div style="display: inline-block; overflow: hidden; height: auto; width: 50%; max-width: 300px; background-color: #f0f0f0; border-radius: 20px; border: 1px solid gray; margin-bottom: 20px;">
            <img src="https://techcrunch.com/wp-content/uploads/2023/09/mistral-7b-v0.1.jpg" style="width: 100%; height: auto; object-fit: contain; border-radius: 20px;">
        </div>
        <h1 style="text-align: center; font-size: 28px; font-weight: bold; color: #333; margin-bottom: 5px;">LLM-Based Search on NS Publications</h1>
        <p style='text-align: center; font-size: small; color: gray; margin-top: 0;'>Powered by © Mistral 7b. This app can make mistakes. Check important info.</p>
    </div>
    """
    st.markdown(custom_html, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_question = st.chat_input("Ask a question about NS publications:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Load FAISS Index Files")
        index_files = st.file_uploader("Upload your FAISS index (.faiss) and metadata (.pkl) files here:", type=["faiss", "pkl"], accept_multiple_files=True)
        if st.button("Load FAISS Index"):
            if index_files and len(index_files) == 2:
                with st.spinner("Loading FAISS index"):
                    faiss_path = ""
                    pkl_path = ""
                    for uploaded_file in index_files:
                        if uploaded_file.name.endswith(".faiss"):
                            faiss_path = save_uploaded_file(uploaded_file, "index_files")
                        elif uploaded_file.name.endswith(".pkl"):
                            pkl_path = save_uploaded_file(uploaded_file, "index_files")

                    if faiss_path and pkl_path:
                        index_folder_path = os.path.dirname(faiss_path)
                        index_file = os.path.splitext(os.path.basename(faiss_path))[0]
                        faiss_index = load_faiss_index(index_folder_path, index_file)
                        st.session_state.conversation = get_conversation_chain(faiss_index)
                        st.success("FAISS index loaded successfully!")
                    else:
                        st.error("Please upload both .faiss and .pkl files.")

if __name__ == '__main__':
    main()
