import validators 
import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate

# Streamlit app configuration
st.set_page_config(page_title="Text Summarization using Langchain")
st.title("Text Summarization using Langchain")
st.subheader('Summarize URL')

# Sidebar for Groq API Key input
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key",value = " " , type='password')

# URL input field
generic_url = st.text_input("URL", label_visibility='collapsed')

# Initialize the language model
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Define the prompt template
prompt_template_str = """ 
Provide a summary of the following content in 300 words:
Content: {text}
"""

# Create a PromptTemplate object
prompt = PromptTemplate(template=prompt_template_str, input_variables=['text'])

# Button to trigger summarization
if st.button("Summarize the Content"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please enter both Groq API Key and URL")
    elif not validators.url(generic_url):
        st.error("Invalid URL")
    else:
        try:
            with st.spinner("Waiting..."):
                # Load documents based on URL type
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(youtube_url=generic_url, add_video_info=True)
                else:
                    st.write("please enter valid youtube url")
                docs = loader.load()  # Call the load method to get the documents
                st.write(docs)
                # Summarization chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # Display the summary
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")
