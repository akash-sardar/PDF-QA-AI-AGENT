"""
Document indexing graph using LangGraph.

This module creates a simple graph that exposes functionality for users to upload
and index documents into a vector store for later retrieval.
"""
import json
import aiofiles
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict


from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

from shared.retrieval import make_retriever
from shared.configuration import ensure_index_configuration, IndexConfigurationAnnotation
from shared.state import IndexStateAnnotation
from utils.main_utils import reduce_docs


async def ingest_docs(state:IndexStateAnnotation, config:Optional[RunnableConfig])->Dict[str, Any]:
    """
    Ingest documents into the vector store.
    
    This function processes documents from the state or loads sample documents
    from a file, then adds them to the configured retriever for indexing.
    
    Args:
        state: Current state containing documents to index
        config: Runtime configuration containing indexing parameters
        
    Returns:
        Dict indicating docs should be deleted from state after processing
        
    Raises:
        ValueError: If no configuration provided or no documents to index
    """    
    if not config:
        raise ValueError("Configuration required to run index_docs")
    
    # Extract configuration parameters
    configuration = ensure_index_configuration(config)
    docs = state.get("docs", [])

    # Load documents from state or sample file
    if not docs or len(docs)==0:
        if configuration["use_sample_docs"]:
            # Load sample documents from JSON file
            async with aiofiles.open(configuration["docs_file"],"r") as f:
                file_content = await f.read()
            serialized_docs = json.loads(file_content)
            docs = reduce_docs([], serialized_docs)
        else:
            raise ValueError("No sample documents found to index")
    else:
        # Process exisitng documents through reducer for consistency
        docs = reduce_docs([], docs)
    
    # Create retriever and add documents to vector store
    retriever = await make_retriever(config)
    await retriever.add_documents(docs)

    # return instruction to clear docs from state
    return {"docs": "delete"}


# Define the graph Structure

builder = StateGraph(
    state_schema=IndexStateAnnotation,
    config_schema = IndexConfigurationAnnotation,
)

# Add the document ingestion node
builder.add_node("ingest_docs", ingest_docs)

# Define the edges and flow
# START -> ingest_docs -> END
builder.add_edge(START, "ingest_docs")
builder.add_edge("ingest_docs", END)

# Compile the graph with Configuration
graph = builder.compile().with_config(run_name="IngestionGraph")





