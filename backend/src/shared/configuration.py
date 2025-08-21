from typing import Dict, Any, Literal, TypedDict
import os
os.makedirs("./src/data", exist_ok=True)
DEFAULT_DOCS_FILE = "./src/data/sample_docs.json"

from langchain_core.runnables import RunnableConfig

class BaseConfigurationAnnotation(TypedDict):
    """
    Configuration Management for indexing and retrieval operations
    parameters:
    retriever_provider: Which vector store to use (only "supabase" supported)
    filter_kwargs: Search filters to apply when retrieving documents
    k: Number of documents to retrieve
    """
    retriever_provider: Literal["supabase"]
    filter_kwargs: Dict[str, Any]
    k:int

def ensure_base_configuration(config: RunnableConfig) -> BaseConfigurationAnnotation:
    """
    Create BaseConfigurationAnnotation from RunnableConfig
    """
    configurable = config.get("configurable", {}) if config else {}

    return BaseConfigurationAnnotation(
        retriever_provider=configurable.get("retriever_provider", "supabase"),
        filter_kwargs=configurable.get("filter_kwargs", {}),
        k = configurable.get("k",5)
    )

class IndexConfigurationAnnotation(BaseConfigurationAnnotation):
    """
    Configuration Management for the indexing process
    """
    docs_file:str
    use_sample_docs:bool

def ensure_index_configuration(config: RunnableConfig) -> IndexConfigurationAnnotation:
   """Create IndexConfigurationAnnotation from RunnableConfig."""
   configurable = config.get("configurable", {}) if config else {}
   base_config = ensure_base_configuration(config)
   
   return IndexConfigurationAnnotation(
       **base_config,
       docs_file=configurable.get("docs_file", DEFAULT_DOCS_FILE),
       use_sample_docs=configurable.get("use_sample_docs", False),
   )
