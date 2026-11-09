import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_documents(hr_policies_path="hr_policies.json",hris_path="hris_employee.json"):
    all_docs = []
    try:
        with open(hr_policies_path,"r") as f:
            policies_data = json.load(f)
            for i in policies_data:
                content = json.dumps(i)
                all_docs.append(
                    Document(page_content=content,
                             metadata={"source":"HR Polices JSON","type":"policy"})
                )

    except FileNotFoundError:
        print(f"Warning : {hr_policies_path} not found")


    try:
            with open(hris_path,"r") as f:
                hris_data = json.load(f)
                for i in hris_data:
                    content = json.dumps(i)
                    all_docs.append(
                        Document(page_content=content,
                                 metadata={"source":"HRIS Employee JSON","type":"employee_data"})
                    )
    
    except FileNotFoundError:
            print(f"Warning : {hris_path} not found")


    spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = spliter.split_documents(all_docs)

    return chunks

    

