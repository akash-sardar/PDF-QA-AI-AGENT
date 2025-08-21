from typing import Annotated, List, Union, Dict, Any
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages

from utils.main_utils import reduce_docs

"""
Represents the state for document indexing and retrieval.
*
This interface defines the structure of the index state, which includes the documents to be indexed and the retriever used for searching these documents.
"""

class IndexStateAnnotation(TypedDict):
   """State for document indexing and retrieval."""
   docs: Annotated[
       List[Document], 
       reduce_docs
   ]

# Type alias for the state
IndexStateType = IndexStateAnnotation


