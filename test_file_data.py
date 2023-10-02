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

# specify the name of the .env file name
env_name = "local.env"  # following example.env template change to your own .env file name
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

# Load certifications.json data file
certification_data_file = open(file="certs_w_services.json", mode="r")
# data_file = open(file="../../DataSet/AzureServices/text-sample_w_embeddings.json", mode="r") # load this file instead if embeddings were previously created and saved.
certification_data = json.load(certification_data_file)
certification_data_file.close()



@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine=embeddings_deployment)
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5)  # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


# Generate embeddings for each certification name, skill name, and service description
for certification in certification_data['certifications']:
    certification_name = certification['certification_name']
    certification_name_embeddings = generate_embeddings(certification_name)
    certification['certificationNameVector'] = certification_name_embeddings
    certification_skills = certification['skills']
    for certification_skill in certification_skills:
        skill_name = certification_skill['skill_name']
        skill_name_embeddings = generate_embeddings(skill_name)
        certification_skill['skill_name_vector'] = skill_name_embeddings
        skill_services = certification_skill['services']
        for skill_service in skill_services:
            service_name = skill_service['service_name']
            service_name_embeddings = generate_embeddings(service_name)
            skill_service['service_name_vector'] = service_name_embeddings
            service_description = skill_service['service_description']
            service_description_embeddings = generate_embeddings(
                service_description)
            skill_service['service_description_vector'] = service_description_embeddings

    certification['@search.action'] = 'upload'


# Save embeddings to sample_text_w_embeddings.json file
with open("certs_w_embeddings_services.json", "w") as f:
    json.dump(certification_data, f)