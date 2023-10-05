import json
import datetime
import time

from azure.core.exceptions import AzureError
from azure.core.credentials import AzureKeyCredential
from azure.cosmos import exceptions, CosmosClient, PartitionKey
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.models import Vector
from azure.search.documents.indexes.models import (
    IndexingSchedule,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchField,
    SearchFieldDataType,
    SearchableField,
    SemanticConfiguration,
    SimpleField,
    PrioritizedFields,
    SemanticField,
    SemanticSettings,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    SearchIndexerDataSourceConnection
)

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

from dotenv import dotenv_values

env_name = "local.env" # following example.env template change to your own .env file name
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

def generate_completion(results):
    system_prompt = '''
    You are an experienced cloud engineer who provides advice to people trying to get hands-on skills while studying for their cloud certifications. You are designed to provide helpful project ideas with a short description, list of possible services to use, and skills that need to be practiced.
    - Only provide project ideas that have products that are part of Microsoft Azure.
    - Each response should be a project idea with a short description, list of possible services to use, and skills that need to be practiced.
    - Write two lines of whitespace between each answer in the list.
    - If you're unsure of an answer, you can say "I don't know" or "I'm not sure" and recommend users search themselves.
    '''

    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for item in results:
        print(item)
        messages.append({"role": "system", "content": item['service_name']})

    response = openai.ChatCompletion.create(engine=completions_deployment, messages=messages)
    
    return response


# Simple function to assist with vector search

def vector_search(query):
    search_client = SearchClient(
        cog_search_endpoint, index_name, cog_search_cred)
    results = search_client.search(
        search_text="",
        vector=Vector(value=generate_embeddings(
            query), k=3, fields="certificationNameVector"),
        select=["certification_name", "service_name", "category"]
    )
    return results


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine=embeddings_deployment)
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI for free tier
    return embeddings

user_input = ""
print("*** Please just ask: Type 'end' to end the session.\n")
user_input = input("Prompt: ")
while user_input.lower() != "end":
    results_for_prompt = vector_search(user_input)
    completions_results = generate_completion(results_for_prompt)
    print("\n")
    print(completions_results['choices'][0]['message']['content'])
    user_input = input("Prompt: ")