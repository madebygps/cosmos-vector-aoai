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

def generate_embeddings(text):
    '''
    Generate embeddings from string to text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine=embeddings_deployment)
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI for free tier
    return embeddings

def create_cosmos_db_client():
    '''
    Create a client to interact with Azure Cosmos DB resource
    '''
    client = CosmosClient(cosmosdb_endpoint, cosmosdb_key)
    return client

def create_cosmos_db_database(client):
    '''
    Create a database in Azure Cosmos DB
    '''
    try:
        database = client.create_database_if_not_exists(id="VectorSearchTutorial")
        print("Created database: {0}".format(database.id))
    except exceptions.CosmosResourceExistsError:
        print("Database already exists: {0}".format(database.id))
    return database

def create_cosmos_db_container(database):
    '''
    Create a container in Azure Cosmos DB
    '''
    try:
        partition_key_path = PartitionKey(path="/id")
        container = database.create_container_if_not_exists(
            id="AzureServices",
            partition_key=partition_key_path
        )
        print("Created container: {0}".format(container.id))
    except exceptions.CosmosResourceExistsError:
        print("Container already exists: {0}".format(container.id))
    return container

def create_cosmos_db_data_items(container, data):
    '''
    Create data items for every entry in the dataset, inset database and container specified above
    '''
    print("Creating data items...")
    for data_item in data:
        try:
            container.create_item(body=data_item)
        except exceptions.CosmosHttpResponseError as e:
            print("Data item already exists in the container: {0}".format(data_item['id']))

def create_search_index():
    '''
    Create a search index and define the schema (names, types, and parameters)
    '''
    print("Creating search index...")
    cog_search_cred = AzureKeyCredential(cog_search_key)
    index_name = "cosmosdb-vector-search-index"
    index_client = SearchIndexClient(endpoint=cog_search_endpoint, credential=cog_search_cred)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String, 
                        searchable = True, retrievable = True),
        SearchableField(name="content", type=SearchFieldDataType.String, 
                        searchable = True, retrievable = True),
        SearchableField(name="category", type=SearchFieldDataType.String,
                        filterable = True, searchable = True, retrievable = True),
        SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable = True, dimensions = 1536, vector_search_configuration="my-vector-config"),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                        searchable = True, dimensions = 1536, vector_search_configuration="my-vector-config"),
    ]

    # Configure vector search
    print("Configuring vector search...")
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

    # Configure semantic search. This will allow us to conduct semantic or hybrid search (with vector search) later
    print("Configuring semantic search...")
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=PrioritizedFields(
            title_field=SemanticField(field_name="title"),
            prioritized_keywords_fields=[SemanticField(field_name="category")],
            prioritized_content_fields=[SemanticField(field_name="content")]
        )
    )

    # Create the semantic settings with the configurations above
    print("Creating semantic settings...")
    semantic_settings = SemanticSettings(configurations=[semantic_config])

    # Create the search index with the semantic settings
    print("Creating search index with semantic settings...")
    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_settings=semantic_settings)
    result = index_client.create_or_update_index(index)
    print(f'{result.name} created.')

def create_indexer():
    '''
    Create an indexer to pull data from Cosmos DB into Cognitive Search
    '''
    print("Creating indexer...")
    cog_search_cred = AzureKeyCredential(cog_search_key)
    ds_name = _create_datasource().name

    indexer = SearchIndexer(
        name="cosmosdb-tutorial-indexer",
        data_source_name=ds_name,
        target_index_name=index_name)

    indexer_client = SearchIndexerClient(endpoint=cog_search_endpoint, credential=cog_search_cred)
    indexer_client.create_or_update_indexer(indexer)

    print("Getting indexer...")
    result = indexer_client.get_indexer("cosmosdb-tutorial-indexer")
    print(result)

def _create_datasource():
    # Here we create a datasource. 
    ds_client = SearchIndexerClient(cog_search_endpoint, cog_search_cred)
    container = SearchIndexerDataContainer(name="AzureServices")
    data_source_connection = SearchIndexerDataSourceConnection(
        name="cosmosdb-tutorial-indexer", type="cosmosdb", connection_string=(f"{cosmosdb_connection_str}Database=VectorSearchTutorial"), container=container
    )
    data_source = ds_client.create_or_update_data_source_connection(data_source_connection)
    return data_source

def run_indexer():
    '''
    Run the indexer we just created
    '''
    print("Running indexer...")
    cog_search_cred = AzureKeyCredential(cog_search_key)
    indexer_client = SearchIndexerClient(endpoint=cog_search_endpoint, credential=cog_search_cred)
    indexer_client.run_indexer("cosmosdb-tutorial-indexer")

def vector_search(query):
    '''
    Simple function to assist with vector search
    '''
    search_client = SearchClient(endpoint=cog_search_endpoint, index_name=index_name, credential=cog_search_cred)
    results = search_client.search(
        search_text="", 
        vector = Vector(value=generate_embeddings(query), k = 3, fields = "contentVector"),
        select=["title", "content", "category"]
        )
    return results

def generate_completion(prompt):
    '''
    Generate completions from OpenAI GPT-3
    '''
    system_prompt = '''
    You are an intelligent assistant for Microsoft Azure services.
    You are designed to provide helpful answers to user questions about Azure services given the information about to be provided.
        - Only answer questions related to the information provided below, provide 3 clear suggestions in a list format.
        - Write two lines of whitespace between each answer in the list.
        - Only provide answers that have products that are part of Microsoft Azure.
        - If you're unsure of an answer, you can say ""I don't know"" or ""I'm not sure"" and recommend users search themselves."
    '''

    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    for item in result:
        messages.append({"role": "system", "content": prompt['content']})

    response = openai.ChatCompletion.create(engine=completions_deployment, messages=messages)
    
    return response

def main():
    # specify the name of the .env file name 
    env_name = "local.env"
    config = dotenv_values(env_name)

    global cosmosdb_endpoint, cosmosdb_key, cosmosdb_connection_str, cog_search_endpoint, cog_search_key, openai.api_type, openai.api_key, openai.api_base, openai.api_version, embeddings_deployment, completions_deployment

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
    data_file = open(file='text-sample.json', mode='r')
    data = json.load(data_file)
    data_file.close()

    # Generate embeddings for title and content fields
    for item in data:
        title = item['title']
        content = item['content']
        title_embeddings = generate_embeddings(title)
        content_embeddings = generate_embeddings(content)
        item['titleVector'] = title_embeddings
        item['contentVector'] = content_embeddings
        item['@search.action'] = 'upload'

    #  Save embeddings to sample_text_w_embeddings.json file
    with open('text-sample_w_embeddings.json', 'w') as f:
        json.dump(data, f)

    # Create the client to interact with Azure Cosmos DB resource
    print("Creating Cosmos DB client...")
    client = create_cosmos_db_client()

    # Create a database
    database = create_cosmos_db_database(client)

    # Create a container in Azure Cosmos DB
    container = create_cosmos_db_container(database)

    # Create data items for every entry in the dataset, inset database and container specified above
    create_cosmos_db_data_items(container, data)

    # Create a search index and define the schema (names, types, and parameters)
    create_search_index()

    # Create an indexer to pull data from Cosmos DB into Cognitive Search
    create_indexer()

    # Run the indexer we just created
    run_indexer()

    # Create a loop of user input and model output. You can now perform Q&A over the sample data!
    user_input = ""
    print("*** Please ask your model questions about Azure services. Type 'end' to end the session.\n")
    user_input = input("Prompt: ")
    while user_input.lower() != "end":
        results_for_prompt = vector_search(user_input)
        completions_results = generate_completion(results_for_prompt)
        print("\n")
        print(completions_results['choices'][0]['message']['content'])
        user_input = input("Prompt: ")