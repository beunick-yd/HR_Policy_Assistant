import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from document_loader import load_and_split_documents
from vector_store import create_vector_store
from rag_chain import build_rag_workflow

load_dotenv()

st.set_page_config(page_title = "HR Policy Assistant", layout = "centered")
st.title("HR Policy Assistant")
st.write("Ask questions about company policies, benefits, or your personal PTO balance.")

@st.cache_resource
def intialize_rag():
    with st.spinner("Indexing HR documents and HRIS data..."):
        chunks = load_and_split_documents()
        vector_store = create_vector_store(chunks)
        workflow=build_rag_workflow(vector_store)
    return workflow

workflow = intialize_rag()

#User Interface

query=st.text_input("Enter your question (e.g., 'How many PTO days does Emma have left?'):")

if(st.button("Submit") and query):
    with st.spinner("Searching policies and employee data...."):
        result = workflow.invoke({
            "messages":[HumanMessage(content=query)]
        })

        finalMsg = result['messages'][-1]
        st.subheader("Answer: ")
        
        if isinstance(finalMsg.content, list):
            for block in finalMsg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    st.write(block["text"])
        else:
        
            st.write(finalMsg.content)