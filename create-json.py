import json


azure_services_file = open("azure-services.json", "r")
certifications_file = open("certifications.json", "r")

azure_services = json.load(azure_services_file)
certifications = json.load(certifications_file)



for certification in certifications['certifications']:
    skills = certification['skills']
    for skill in skills:
        for service in skill['services']:
            for azure_service in azure_services:
                if azure_service['title'] == service:
                    service_description = azure_service['content']
                    break
            service_dict = {
                "service_name": service,
                "service_description": service_description
            }
            skill['services'] = [service_dict if s == service else s for s in skill['services']]
            
# Save to new file
with open("certs_w_services.json", "w") as f:
    json.dump(certifications, f)

