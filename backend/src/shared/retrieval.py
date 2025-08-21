import os
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.runnables import RunnableConfig

from supabase import create_client

from shared.configuration import BaseConfigurationAnnotation, ensure_base_configuration

async def make_supabase_retriever(configuration: BaseConfigurationAnnotation)-> VectorStoreRetriever:
    try:
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_SERVICE_ROLE_KEY"):
            raise ValueError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not found in environment variables")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        supabase_client = create_client(
            os.getenv("SUPABASE_URL", ""),
            os.getenv("SUPABASE_SERVICE_ROLE_KEY","")
        )

        vector_store = SupabaseVectorStore(
            embedding=embeddings,
            client=supabase_client,
            table_name="documents",
            query_name="match_documents"
        )

        return vector_store.as_retriever(
            search_kwargs = {
                "k":configuration.k,
                "filter":configuration.filter_kwargs
            }
        )
    except Exception as e:
        print(f"Error encountered while retrieving: {e}")

async def make_retriever(config: RunnableConfig) -> VectorStoreRetriever:
    try:
        configuration = ensure_base_configuration(config)
        
        if configuration.retriever_provider == "supabase":
            return await make_supabase_retriever(configuration)
        else:
            raise ValueError(f"Unsupported retriever provider: {configuration.retriever_provider}")
        
    except Exception as e:
        print(f"Error encountered while making retriever: {e}")