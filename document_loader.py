import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_documents(hr_policies_path="hr_policies.json",hris_path="hris_employee.json"):
    all_docs = []
    
    file_configs = [
        (hr_policies_path, "HR Polices JSON", "policy"),
        (hris_path, "HRIS Employee JSON", "employee_data")
    ]
    for file_path, source_name,doc_type in file_configs:
         
        try:
            with open(file_path,"r") as f:
                policies_data = json.load(f)
                for i in policies_data:
                    content = json.dumps(i)
                    all_docs.append(
                        Document(page_content=content,
                             metadata={"source":source_name,"type":doc_type})
                        )

        except FileNotFoundError:
            print(f"Warning : {file_path} not found")
    

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    return chunks

    

