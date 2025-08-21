from typing import List, Union, Dict, Any, Optional
import uuid

from langchain_core.documents import Document

def reduce_docs(
        existing: Optional[List[Document]] = None,
        new_docs: Optional[Union[List[Document], List[Dict[str, Any]], List[str], str]] = None,
)-> List[Document]:
    """
    Reduces document array based on new documents or actions
    """
    # delete all documents if "delete" command is recieved
    if new_docs == "delete":
        return []
    
    existing_list = existing or []
    existing_ids = {doc.metadata.get("uuid") for doc in existing_list if doc.metadata}

    # If new document recieved is a string, add it to exisitng document and return entire list
    if isinstance(new_docs, str):
        doc_id = str(uuid.uuid4())
        return [
            *existing_list,
            Document(page_content=new_docs, metadata = {"uuid":doc_id})
        ]
    
    # If new document is a list of documents
    new_list: List[Document] = []
    if isinstance(new_docs, list):
        for item in new_docs:
            if isinstance(item, str):
                item_id = str(uuid.uuid4())
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.append(item_id)
            elif isinstance(item, dict):
                metadata = getattr(item, 'metadata', {}) if hasattr(item, "metadata") else {}
                item_id = metadata.get("uuid", str(uuid.uuid4()))

                if item_id not in existing_ids:
                    if "page_content" in item:
                        new_list.append(Document(
                            page_content=item["page_content"],
                            metadata={**metadata, "uuid":item_id}
                        ))
                    else:
                        new_list.append(Document(
                            page_content="",
                            metadata={**item, "uuid":item_id}       
                        ))
                    existing_ids.add(item_id)                 
    return [*existing_list, *new_list]


    