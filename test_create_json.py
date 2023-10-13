import json
import unittest
from create_json import add_service_details

class TestCreateJson(unittest.TestCase):
    def setUp(self):
        # Load test data
        with open("test_data.json", "r") as f:
            self.test_data = json.load(f)

    def test_add_service_details(self):
        # Call the function with test data
        certifications = self.test_data["certifications"]
        azure_services = self.test_data["azure_services"]
        add_service_details(certifications, azure_services)

        # Check that the service descriptions were added correctly
        expected_output = self.test_data["expected_output"]
        self.assertEqual(certifications, expected_output)

if __name__ == '__main__':
    unittest.main()