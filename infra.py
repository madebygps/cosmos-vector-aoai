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

# Load text-sample.json data file
data_file = open(file="text-sample.json", mode="r")
#data_file = open(file="../../DataSet/AzureServices/text-sample_w_embeddings.json", mode="r") # load this file instead if embeddings were previously created and saved.
data = json.load(data_file)
data_file.close()


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

# Generate embeddings for title and content fields
# for item in data:
#     title = item['title']
#     content = item['content']
#     title_embeddings = generate_embeddings(title)
#     content_embeddings = generate_embeddings(content)
#     item['titleVector'] = title_embeddings
#     item['contentVector'] = content_embeddings
#     item['@search.action'] = 'upload'

# # Save embeddings to sample_text_w_embeddings.json file
# with open("text-sample_w_embeddings.json", "w") as f:
#     json.dump(data, f)

# Create the client to interact with the Azure Cosmos DB resource
client = CosmosClient(cosmosdb_endpoint, cosmosdb_key)

# Create a database in Azure Cosmos DB.
try:
    database = client.create_database_if_not_exists(id="VectorSearchTutorial")
    print(f"Database created: {database.id}")

except exceptions.CosmosResourceExistsError:
    print("Database already exists.")

# Create a container in Azure Cosmos DB.
try:
    partition_key_path = PartitionKey(path="/id")
    container = database.create_container_if_not_exists(
        id="AzureServices",
        partition_key=partition_key_path
    )
    print(f"Container created: {container.id}")

except exceptions.CosmosResourceExistsError:
    print("Container already exists.")

# Create data items for every entry in the dataset, insert them into the database and collection specified above.
for data_item in data:
    try:
        container.create_item(body=data_item)
    
    except exceptions.CosmosResourceExistsError:
        print("Data item already exists.")

# Create index

cog_search_cred = AzureKeyCredential(cog_search_key)
index_name = "cosmosdb-vector-search-index"

# Create a search index and define the schema (names, types, and parameters)
index_client = SearchIndexClient(
    endpoint=cog_search_endpoint, credential=cog_search_cred)
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="title", type=SearchFieldDataType.String,
                    searchable=True, retrievable=True),
    SearchableField(name="content", type=SearchFieldDataType.String,
                    searchable=True, retrievable=True),
    SearchableField(name="category", type=SearchFieldDataType.String,
                    filterable=True, searchable=True, retrievable=True),
    SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, dimensions=1536, vector_search_configuration="my-vector-config"),
    SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, dimensions=1536, vector_search_configuration="my-vector-config"),
]

# Configure vector search
vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="my-vector-config",
            kind="hnsw",
            hnsw_parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 1000,
                "metric": "cosine"
            }
        )
    ]
)

# Configure semantic search. This will allow us to conduct semantic or hybrid search (with vector search) later on if desired.
semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(
        title_field=SemanticField(field_name="title"),
        prioritized_keywords_fields=[SemanticField(field_name="category")],
        prioritized_content_fields=[SemanticField(field_name="content")]
    )
)

# Create the semantic settings with the configuration
semantic_settings = SemanticSettings(configurations=[semantic_config])

# Create the search index with the semantic settings
index = SearchIndex(name=index_name, fields=fields,
                    vector_search=vector_search, semantic_settings=semantic_settings)
result = index_client.create_or_update_index(index)
print(f' {result.name} created')

# Create indexer

def _create_datasource():
    # Here we create a datasource. 
    ds_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
    container = SearchIndexerDataContainer(name="AzureServices")
    data_source_connection = SearchIndexerDataSourceConnection(
        name="cosmosdb-tutorial-indexer", type="cosmosdb", connection_string=(f"{cosmosdb_connection_str}Database=VectorSearchTutorial"), container=container
    )
    data_source = ds_client.create_or_update_data_source_connection(data_source_connection)
    return data_source

ds_name = _create_datasource().name

indexer = SearchIndexer(
        name="cosmosdb-tutorial-indexer",
        data_source_name=ds_name,
        target_index_name=index_name)

indexer_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
indexer_client.create_or_update_indexer(indexer)  # create the indexer

result = indexer_client.get_indexer("cosmosdb-tutorial-indexer")
print(result)

# Run the indexer we just created.
indexer_client.run_indexer(result.name)

# Simple function to assist with vector search
def vector_search(query):
    search_client = SearchClient(cog_search_endpoint, index_name, cog_search_cred)  
    results = search_client.search(  
        search_text="",  
        vector=Vector(value=generate_embeddings(query), k=3, fields="contentVector"),  
        select=["title", "content", "category"] 
    )
    return results

query = "tools for software development"  
results = vector_search(query)
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"Content: {result['content']}")  
    print(f"Category: {result['category']}\n")  