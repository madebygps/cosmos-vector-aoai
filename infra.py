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
    ComplexField,
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
env_name = "local.env"
config = dotenv_values(env_name)

cosmosdb_endpoint = config['cosmos_db_api_endpoint']
cosmosdb_key = config['cosmos_db_api_key']
cosmosdb_connection_str = config['cosmos_db_connection_string']

cog_search_endpoint = config['cognitive_search_api_endpoint']
cog_search_key = config['cognitive_search_api_key']
cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "project-generator-index"

openai.api_type = config['openai_api_type']
openai.api_key = config['openai_api_key']
openai.api_base = config['openai_api_endpoint']
openai.api_version = config['openai_api_version']
embeddings_deployment = config['openai_embeddings_deployment']
completions_deployment = config['openai_completions_deployment']

# Load certifications.json data file
certification_data_file = open(file="certs_services_id.json", mode="r")
certification_data = json.load(certification_data_file)
certification_data_file.close()


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Embeddings are representations of semantic meaning of text, stored in a vector.  
    We use Azure Open AI ada model to generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine=embeddings_deployment)
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5)  # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


# # Generate embeddings for each certification name, skill name, service name and service description

for certification in certification_data:
    certification_name = certification['certification_name']
    certification_name_embeddings = generate_embeddings(certification_name)
    certification['certificationNameVector'] = certification_name_embeddings
    certification_service = certification['service_name']
    certification_service_embeddings = generate_embeddings(
        certification_service)
    certification['certificationServiceVector'] = certification_service_embeddings
    # this will be used by Azure Cognitive Search to mark each certification to be uploaded to the index so it can be searched
    certification['@search.action'] = 'upload'


# # Save embeddings to new file
with open("certs_embeddings.json", "w") as f:
    json.dump(certification_data, f)

# Create the client to interact with the Azure Cosmos DB resource
client = CosmosClient(cosmosdb_endpoint, cosmosdb_key)

# Create a database in Azure Cosmos DB.
try:
    database = client.create_database_if_not_exists(id="CertificationData")
    print(f"Database created: {database.id}")

except exceptions.CosmosResourceExistsError:
    print("Database already exists.")

# Create a container in Azure Cosmos DB.
try:
    partition_key_path = PartitionKey(path="/id")
    container = database.create_container_if_not_exists(
        id="Certifications",
        partition_key=partition_key_path
    )
    print(f"Container created: {container.id}")

except exceptions.CosmosResourceExistsError:
    print("Container already exists.")

# Upload each certification to the cosmos db container
for certification_data_item in certification_data:
    try:
        container.create_item(body=certification_data_item)

    except exceptions.CosmosResourceExistsError:
        print("Data item already exists.")


# # Create a search index and define the schema (names, types, and parameters)
# # A search index is like a database of searchable content.
index_client = SearchIndexClient(
    endpoint=cog_search_endpoint, credential=cog_search_cred)
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="certification_name", type=SearchFieldDataType.String,
                    searchable=True, retrievable=True),
    SearchableField(name="service_name", type=SearchFieldDataType.String,
                    searchable=True, retrievable=True),
    SearchableField(name="category", type=SearchFieldDataType.String,
                    searchable=True, retrievable=True),
    SearchField(name="certificationNameVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, dimensions=1536, vector_search_configuration="my-vector-config"),
    SearchField(name="certificationServiceVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, dimensions=1536, vector_search_configuration="my-vector-config")
]

# https://azuresdkdocs.blob.core.windows.net/$web/python/azure-search-documents/latest/index.html#creating-an-index

# Configure vector search for Azure Cognitive Search Index.
# Vector search is used to find similar items based on their embeddings.
# This enables indexing, storing, and retrieivng of embeddings in a search index.
# ANN approximate nearest neighbor search is a class of algorithms designed for finding matches in a vector space.
# Hierarchical Navigable Small Worlds HSWN is one of the best performing algorithms for ANN searh
# https://www.pinecone.io/learn/series/faiss/hnsw/
# https://learn.microsoft.com/en-us/azure/search/vector-search-overview
# https://www.youtube.com/@jamesbriggs/videos
vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw",  # hierarchical navigable small world
            hnsw_parameters={
                "m": 5,  # number of bi-directional links created for each element during construction
                "efConstruction": 400,  # number of elements to be visited during construction
                "efSearch": 1000,  # number of elements to visited during search
                "metric": "cosine"  # distance metric used to compute the similarity between two vectors
            }
        )
    ]
)

# Configure semantic search. This will allow us to conduct semantic or hybrid search (with vector search) later on if desired.
# A semantic configuration defines the fields that will be used for semantic search.
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="certification_name"),
        prioritized_keywords_fields=[SemanticField(field_name="category")],
        prioritized_content_fields=[SemanticField(field_name="service_name")]

    )
)

# Create the semantic settings with the configuration
# Create the search index with the semantic settings
semantic_settings = SemanticSettings(configurations=[semantic_config])

index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search, semantic_settings=semantic_settings)
result = index_client.create_or_update_index(index)
print(f' {result.name} created')

# Create indexer

# create a datasource for the indexer. In this case it's cosmosdb.
# a datasource is where the search index can get its data from.


def _create_datasource():
    # Here we create a datasource.
    ds_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
    container1 = SearchIndexerDataContainer(name="Certifications")
    data_source_connection = SearchIndexerDataSourceConnection(
        name="project-indexer", type="cosmosdb", connection_string=(f"{cosmosdb_connection_str}Database=CertificationData"), container=container1
    )
    data_source = ds_client.create_or_update_data_source_connection(
        data_source_connection)
    return data_source


ds_name = _create_datasource().name

# create the indexer. An indexer will take the data from the datasource and put it into the search index.
indexer = SearchIndexer(
    name="project-generator-indexer",
    data_source_name=ds_name,
    target_index_name=index_name)
indexer_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
indexer_client.create_or_update_indexer(indexer)  # create the indexer
result = indexer_client.get_indexer("project-generator-indexer")
print(f"{result.name} created")

# Run the indexer we just created.
indexer_client.run_indexer(result.name)