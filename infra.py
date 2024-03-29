import json
import time
import azure.cosmos.exceptions as exceptions

from azure.core.credentials import AzureKeyCredential
from azure.cosmos import CosmosClient, PartitionKey
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
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
from openai import AzureOpenAI


from tenacity import retry, wait_random_exponential, stop_after_attempt

from dotenv import dotenv_values

# specify the name of the .env file name
env_name = ".env"
config = dotenv_values(env_name)

client = AzureOpenAI(azure_endpoint=config['openai_api_endpoint'], api_key=config['openai_api_key'],
api_version=config['openai_api_version'])

cosmosdb_endpoint = config['cosmosdb_endpoint']
cosmosdb_key = config['cosmosdb_key']
cosmosdb_connection_str = config['cosmosdb_connection_string']

cog_search_endpoint = config['cognitive_search_api_endpoint']
cog_search_key = config['cognitive_search_api_key']
cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "project-generator-idea-index"

# TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base=config['openai_api_endpoint'])'
# openai.api_base = config['openai_api_endpoint']
embeddings_deployment = config['openai_embeddings_deployment']
completions_deployment = config['openai_completions_deployment']

# Load certifications.json data file
certification_data_file = open(file="certs_services_id.json", mode="r")
certification_data = json.load(certification_data_file)
certification_data_file.close()



@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    """
    Generates embeddings for the given text using the specified engine.

    Args:
        text (str): The input text for which embeddings need to be generated.

    Returns:
        list: A list of embeddings generated for the input text.
    """
    response = client.embeddings.create(input=[text], model=embeddings_deployment)
    embeddings = response.data[0].embedding
    time.sleep(0.5)  # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


# Generate embeddings for each certification name, skill name, service name and service description
for certification in certification_data:
    certification_name = certification['certification_name']
    certification_name_embeddings = generate_embeddings(certification_name)
    certification['certificationNameVector'] = certification_name_embeddings
    certification_service = certification['service_name']
    certification_service_embeddings = generate_embeddings(
        certification_service)
    certification['certificationServiceVector'] = certification_service_embeddings
    # Marks each certification to be uploaded to the index so it can be searched
    certification['@search.action'] = 'upload'


# Save embeddings to new file
with open("certs_embeddings.json", "w") as f:
    json.dump(certification_data, f)

# Create the client to interact with the Azure Cosmos DB resource
client = CosmosClient(cosmosdb_endpoint, cosmosdb_key)

# Create a database in Azure Cosmos DB.
try:
    database = client.create_database_if_not_exists(id="CertificationProjectData")
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


# Create search index schema (names, types, and parameters)
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


# Configure vector search for Azure Cognitive Search Index.
vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw", 
            hnsw_parameters={
                "m": 5,  
                "efConstruction": 400,
                "efSearch": 1000,  
                "metric": "cosine" 
            }
        )
    ]
)

# Configure semantic search. 
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="certification_name"),
        prioritized_keywords_fields=[SemanticField(field_name="category")],
        prioritized_content_fields=[SemanticField(field_name="service_name")]

    )
)

# Create the semantic settings with the configuration
semantic_settings = SemanticSettings(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search, semantic_settings=semantic_settings)
result = index_client.create_or_update_index(index)
print(f' {result.name} created')


# Create a datasource for the indexer. In this case it's cosmosdb.
def _create_datasource():
    ds_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
    container1 = SearchIndexerDataContainer(name="Certifications")
    data_source_connection = SearchIndexerDataSourceConnection(
        name="project-indexer", type="cosmosdb", connection_string=(f"{cosmosdb_connection_str}Database=CertificationProjectData"), container=container1
    )
    data_source = ds_client.create_or_update_data_source_connection(
        data_source_connection)
    return data_source

ds_name = _create_datasource().name

# Create the indexer
indexer = SearchIndexer(
    name="project-generator-indexer",
    data_source_name=ds_name,
    target_index_name=index_name)
indexer_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
indexer_client.create_or_update_indexer(indexer)
result = indexer_client.get_indexer("project-generator-indexer")
print(f"{result.name} created")

# Run the indexer we just created
indexer_client.run_indexer(result.name)