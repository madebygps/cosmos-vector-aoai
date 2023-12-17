import uuid
import json

# Load the JSON data from the file
with open('ai102.json', 'r') as f:
    data = json.load(f)

# Iterate through the array and add a unique id property to each object
for obj in data:
    obj['id'] = str(uuid.uuid4())

# Write the updated JSON data back to the file
with open('ai102.json', 'w') as f:
    json.dump(data, f, indent=2)

print(data)