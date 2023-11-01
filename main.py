import os
import time
import openai
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding
from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
from semantic_kernel.core_skills import TextMemorySkill
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import dotenv_values

env_name = ".env"
config = dotenv_values(env_name)

cosmosdb_endpoint = config['cosmos_db_api_endpoint']
cosmosdb_key = config['cosmos_db_api_key']
cosmosdb_connection_str = config['cosmos_db_connection_string']
cog_search_endpoint = config['cognitive_search_api_endpoint']
cog_search_key = config['cognitive_search_api_key']
openai.api_type = config['openai_api_type']
openai.api_key = config['openai_api_key']
openai.api_base = config['openai_api_endpoint']
openai.api_version = config['openai_api_version']
embeddings_deployment = config['openai_embeddings_deployment']
completions_deployment = config['openai_completions_deployment']
cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "project-generator-index"

kernel = sk.Kernel()
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service("chat_completion", AzureChatCompletion(
    deployment, endpoint, api_key))

# Configure embeddings service
kernel.add_text_embedding_generation_service(
    "ada",
    AzureTextEmbedding(
        "embeddings-v0",
        config['openai_api_endpoint'],
        config['openai_api_key']
    ),
)

# Add memory
kernel.register_memory_store(
    memory_store=AzureCognitiveSearchMemoryStore(
        1536, cog_search_endpoint, cog_search_key
    )
)


# Add skills
skill = kernel.import_semantic_skill_from_directory("skills", "GenerateSkill")
generator_function = skill["Project"]
kernel.import_skill(TextMemorySkill())


def generate_completion(user_input):
    context = kernel.create_new_context()
    context[TextMemorySkill.COLLECTION_PARAM] = "project-generator-index"
    response = generator_function(user_input, context=context)
    return response


async def vector_search(query):
    """
    Searches for results in the specified index using the provided query and returns the top 3 results.

    Args:
        query (str): The query to search for.

    Returns:
        list: A list of dictionaries containing the top 3 results, each with the keys "certification_name", "service_name", and "category".
    """

    results = await kernel.memory.search_async("project-generator-index", query, limit=3)
    print(results)

    return results




async def main():
    user_input = ""
    print("*** Type 'end' to end the session.\n")
    user_input = input("Certification Name: ")
    while user_input.lower() != "end":
        await vector_search(user_input)
        completions_results = generate_completion(user_input)
        print("\n")
        print(completions_results)
        user_input = input("Certification Name: ")

asyncio.run(main())
