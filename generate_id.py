import json
import uuid


# Load certifications.json data file
certification_data_file = open(file="certs_w_services.json", mode="r")
# data_file = open(file="../../DataSet/AzureServices/text-sample_w_embeddings.json", mode="r") # load this file instead if embeddings were previously created and saved.
certification_data = json.load(certification_data_file)
certification_data_file.close()

# Add an 'id' field to each item in the JSON data
for item in certification_data['certifications']:
    item['id'] = str(uuid.uuid4())


# Save embeddings to sample_text_w_embeddings.json file
with open("certs_w_embeddings_services_and_ids.json", "w") as f:
    json.dump(certification_data, f)