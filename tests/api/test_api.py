import unittest

from fastapi.testclient import TestClient
from challenge import application


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(application)

    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        # Caminho para o arquivo CSV real no repositório
        file_path = '/mnt/c/Users/Datum TI/dev/ml-engineer-latam-chalenge/data/data.csv'

        # Fazer a requisição de teste abrindo o arquivo real
        with open(file_path, 'rb') as f:
            response_train = self.client.post("/train", files={'file': ('data.csv', f, 'text/csv')})
            self.assertEqual(response_train.status_code, 200)
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})


    def test_should_failed_unkown_column_1(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self):
        data = {
            "flights": [
                {
                    "OPERA": "Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)