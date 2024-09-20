from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.search import LocalSearch
from transformers.agents import Tool


class GraphLocalRetrieverTool(Tool):
    name = "graphLocalRetriever"
    description = "Using graphRAG local search, it is designed to perform in-depth analysis, identify relationships, or make decisions based on comparisons across various topics and to find similarities, contrast different elements, or classify data based on defined criteria, ComparativeQueryAnalyzer efficiently processes and delivers relevant, insightful results tailored to your query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target context. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, localSearch: LocalSearch, **kwargs):
        super().__init__(**kwargs)
        self.localSearch = localSearch

    async def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        result = await self.localSearch.asearch(query)

        return result
    
class GraphGlobalRetrieverTool(Tool):
    name = "graphGlobalRetriever"
    description = "Using graphRAG global search, it is designed to answer high-level, strategic questions: how to successfully complete the game or looking to understand the core themes and narrative."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target context. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, globalSearch:GlobalSearch, **kwargs):
        super().__init__(**kwargs)
        self.globalSearch = globalSearch

    async def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        result = await self.globalSearch.asearch(query)

        return result.response
# TODO llama_index index definition is not fillfuled with llama_index
# class BaselineRagRetrieverTool(Tool):
#     import logging
#     import sys

#     from azure.core.credentials import AzureKeyCredential
#     from azure.search.documents import SearchClient
#     from azure.search.documents.indexes import SearchIndexClient
#     from IPython.display import Markdown, display
#     from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
#     from llama_index.core.settings import Settings
#     from llama_index.core.vector_stores.types import VectorStoreQueryMode
#     from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
#     from llama_index.llms.azure_openai import AzureOpenAI
#     from llama_index.vector_stores.azureaisearch import (
#         AzureAISearchVectorStore,
#         IndexManagement,
#         MetadataIndexFieldType,
#     )

#     llm = AzureOpenAI(
#         model="gpt-4o",
#         deployment_name="gpt-4o",
#         api_key="41433befdf824b348136ef86bb897073",
#         azure_endpoint="https://openai-hu-non-product-test.openai.azure.com/",
#         api_version="2024-02-15-preview",
#     )

#     # You need to deploy your own embedding model as well as your own chat completion model
#     embed_model = AzureOpenAIEmbedding(
#         model="text-embedding-ada-002",
#         deployment_name="text-embedding-ada-002",
#         api_key="41433befdf824b348136ef86bb897073",
#         azure_endpoint="https://openai-hu-non-product-test.openai.azure.com/",
#         api_version="2024-02-15-preview",
#     )

#     Settings.llm = llm
#     Settings.embed_model = embed_model
    
#     search_service_api_key = "je3yK1ohrjF2QYUZ0w4jv3GUiKl3rAExNOFwircuHhAzSeB0dHTB"
#     search_service_endpoint = "https://ai-search-hu-west-us-3.search.windows.net"
#     search_service_api_version = "2023-11-01"
#     credential = AzureKeyCredential(search_service_api_key)


#     # Index name to use
#     index_name = "baseline-rag-index01"

#     # Use index client to demonstrate creating an index
#     index_client = SearchIndexClient(
#         endpoint=search_service_endpoint,
#         credential=credential,
#     )

#     # Use search client to demonstration using existing index
#     search_client = SearchClient(
#         endpoint=search_service_endpoint,
#         index_name=index_name,
#         credential=credential,
#     )

#     metadata_fields = {
#     "title": "title",
#     "filepath": ("filepath", MetadataIndexFieldType.STRING),
#     "url": "url"}

#     vector_store = AzureAISearchVectorStore(
#         search_or_index_client=search_client,
#         index_management=IndexManagement.VALIDATE_INDEX,
#         id_field_key="id",
#         chunk_field_key="content",
#         embedding_field_key="contentVector",
#         embedding_dimensionality=1536,
#         language_analyzer="zh-Hans",
#         doc_id_field_key="chunk_id",
#     )

#     storage_context = StorageContext.from_defaults(vector_store=vector_store)
#     index = VectorStoreIndex.from_documents(
#         [],
#         storage_context=storage_context,
#     )

#     hybrid_retriever = index.as_retriever(vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID)



#     name = "baselineRagRetriever"
#     description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
#     inputs = {
#         "query": {
#             "type": "text",
#             "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
#         }
#     }
#     output_type = "text"



#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def forward(self, query: str) -> str:
#         assert isinstance(query, str), "Your search query must be a string"

#         docs = self.hybrid_retriever.retrieve(query)

#         return "\nRetrieved documents:\n" + "".join(
#             [f"===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
#         )


class BaselineRagRetrieverTool(Tool):

    import os

    from azure.identity import DefaultAzureCredential
    from dotenv import load_dotenv

    load_dotenv(override=True) # take environment variables from .env.

    # Variables not used here do not need to be updated in your .env file
    endpoint = "https://ai-search-hu-west-us-3.search.windows.net"
    key_credential = "je3yK1ohrjF2QYUZ0w4jv3GUiKl3rAExNOFwircuHhAzSeB0dHTB"
    index_name = "baseline-rag-index01"
    azure_openai_endpoint = "https://openai-hu-non-product-test.openai.azure.com/"
    azure_openai_key = "41433befdf824b348136ef86bb897073"
    azure_openai_embedding_deployment = "text-embedding-ada-002"
    azure_openai_api_version = "2024-02-15-preview"

    credential = key_credential or DefaultAzureCredential()
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from langchain_openai import AzureOpenAIEmbeddings

    openai_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(openai_credential, "https://cognitiveservices.azure.com/.default")

    # Use API key if provided, otherwise use RBAC authentication
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=azure_openai_embedding_deployment,
        openai_api_version=azure_openai_api_version,
        azure_endpoint=azure_openai_endpoint,
        api_key=azure_openai_key,
        azure_ad_token_provider=token_provider if not azure_openai_key else None)

    os.environ["AZURESEARCH_FIELDS_CONTENT_VECTOR"] = "contentVector"   
    from langchain.vectorstores.azuresearch import AzureSearch

    vector_store = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key_credential,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
        semantic_configuration_name="default"
    )

    name = "baselineRagRetriever"
    description = "Using semantic similarity, it is not designed to answer high-level, strategic questions: how to successfully complete the game or looking to understand the core themes and narrative. And it is not designed to perform in-depth analysis, identify relationships, or make decisions based on comparisons across various topics and to find similarities, contrast different elements, or classify data based on defined criteria, ComparativeQueryAnalyzer efficiently processes and delivers relevant, insightful results tailored to your query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"
  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vector_store.semantic_hybrid_search_with_score(query=query,k=10)

        docs_and_scores = docs[:3]

        result = "\nRetrieved documents:\n"
        index = 1
        for doc, score in docs_and_scores:  
            print("-" * 80) 
            answers = doc.metadata['answers']  
            if answers:  
                if answers.get('highlights'):  
                    print(f"Semantic Answer: {answers['highlights']}")  
                else:  
                    print(f"Semantic Answer: {answers['text']}")  
                print(f"Semantic Answer Score: {score}")  
            print("page_content:", doc.page_content)
            result.join(doc.page_content) 
            captions = doc.metadata['captions']
            print(f"Score: {score}") 
            if captions:  
                if captions.get('highlights'):  
                    print(f"Caption: {captions['highlights']}")  
                else:  
                    print(f"Caption: {captions['text']}")  
            else:  
                print("Caption not available")  

        result = "\nRetrieved documents:\n" + "".join(
            [f"===== Document {str(i)} =====\n" + doc.page_content for  i, (doc, score) in enumerate(docs_and_scores)]
        )

        return result
    

class WebSearchRetrieverTool(Tool):
    name = "webSearchRetriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "text"

    def __init__(self, bingSearch, **kwargs):
        super().__init__(**kwargs)
        self.bingSearch = bingSearch

    def forward(self, query: str) -> str:
        pass