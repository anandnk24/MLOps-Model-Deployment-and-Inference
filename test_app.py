import os
from fastapi.testclient import TestClient
from app.server import app

os.chdir('app')
client = TestClient(app)

""" 
We've built our web application, and containerized it with Docker.
But imagine a team of ML engineers and scientists that needs to maintain, improve and scale this service over time. 
It would be nice to write some tests to ensure we don't regress! 

  1. `Pytest` is a popular testing framework for Python. Check out https://docs.pytest.org/en/7.1.x/getting-started.html 
   
  2. How do we test FastAPI applications with Pytest? 
    (i) Introduction to testing FastAPI: https://fastapi.tiangolo.com/tutorial/testing/
    (ii) Testing FastAPI with startup and shutdown events: https://fastapi.tiangolo.com/advanced/testing-events/

"""

def test_root():
    """
    Test the root ("/") endpoint, which just returns a {"Hello": "World"} json response
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_predict_empty():
    """
    Test the "/predict" endpoint, with an empty request body; This is an unprocessable request
    """
    with TestClient(app) as client:
        response = client.post("/predict", headers={"accept": "application/json", "Content-Type": "application/json"}, data="{}")
        assert response.status_code == 422  


def test_predict_en_lang():
    """
    Test the "/predict" endpoint, with an input text in English (you can use one of the test cases provided in README.md)
    """
    input = {
        "source": "BBC Technology", \
        "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm", \
        "title": "System gremlins resolved at HSBC", \
        "description": "Computer glitches which led to chaos for HSBC customers on Monday are fixed, the High Street bank confirms."
    }
    with TestClient(app) as client:
        response = client.post("/predict", headers={"accept": "application/json", "Content-Type": "application/json"}, json=input)
        assert response.status_code == 200
        assert response.json()["label"] == "Sci/Tech"



def test_predict_es_lang():
    """
    Test the "/predict" endpoint, with an input text in Spanish. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error? (... It returns an error!)
    """
    input = {
        "source": "BBC Tecnología", \
        "url": "", \
        "title": "Gremlins del sistema resueltos en HSBC", \
        "description": "Los fallos informáticos que llevaron al caos a los clientes de HSBC el lunes se solucionaron, confirma el banco High Street."
    }
    with TestClient(app) as client:
        response = client.post("/predict", headers={"accept": "application/json", "Content-Type": "application/json"}, json=input)
        assert response.status_code == 200
        assert response.json()["label"] == "Sci/Tech"



def test_predict_non_ascii():
    """
    [TO BE IMPLEMENTED]
    Test the "/predict" endpoint, with an input text that has non-ASCII characters. 
    Does the tokenizer and classifier handle this case correctly? Does it return an error? (... Seems like it handles it correctly!)
    """
    input = {
        "source": "BBC Technology", \
        "url": "http://news.bbc.co.uk/go/click/rss/0.91/public/-/2/hi/business/4144939.stm", \
        "title": "System gremlins resolved at HSBC", \
        "description": "C∅mpute® glitc™es which¿ led to ch‰os for HS฿Ĉ customərs on Monday are fixed, the Æigh §treet bank confirms."
    }
    with TestClient(app) as client:
        response = client.post("/predict", headers={"accept": "application/json", "Content-Type": "application/json"}, json=input)
        assert response.status_code == 200
        assert response.json()["label"] == "Sci/Tech"