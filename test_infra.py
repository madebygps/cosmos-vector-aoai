import unittest
from unittest.mock import patch, MagicMock

from infra import generate_embeddings


class TestInfra(unittest.TestCase):
    @patch('infra.openai.Embedding.create')
    def test_generate_embeddings(self, mock_create):
        mock_response = MagicMock()
        mock_response.return_value = {
            'data': [{'embedding': [1, 2, 3]}]
        }
        mock_create.return_value = mock_response

        text = 'test text'
        embeddings = generate_embeddings(text)

        self.assertEqual(embeddings, [1, 2, 3])
        mock_create.assert_called_once_with(input=text, engine='your_engine_name')