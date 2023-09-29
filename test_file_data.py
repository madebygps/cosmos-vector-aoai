# Load text-sample.json data file
import json


certifications_data_file = open(file="certifications.json", mode="r")
#data_file = open(file="../../DataSet/AzureServices/text-sample_w_embeddings.json", mode="r") # load this file instead if embeddings were previously created and saved.
certifications_data = json.load(certifications_data_file)
certifications_data_file.close()

# Load services data
services_data_file = open(file="azure-services.json", mode="r")
services_data = json.load(services_data_file)
services_data_file.close()


for certification in certification_data['certifications']:
    certification_skills = certification['skills']
    for skill in certification_skills:
        certification_skills_embeddings = generate_embeddings(skill)
        skill_name = skill['skill_name']
        print(skill_name)
        sub_skills = skill['sub_skills']
        for sub_skill in sub_skills:
            sub_skill_name = sub_skill
            print(f" - {sub_skill_name}")
        services = skill['services']
        for service in services:
            service_name = service
            service_description = None
            for service_item in services_data:
                if service_item['title'] == service_name:
                    service_description = service_item['content']
                    break
            print(f" -- {service_name}")