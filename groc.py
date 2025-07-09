import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class GrocClient:
    def __init__(self, api_key, model="llama3-8b-8192", base_url="https://api.groq.com/openai/v1"):  # official Groq API endpoint
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip('/')

        # Initialiser la session avec logique de retry
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def embed(self, text: str):
        """Obtenir des embeddings depuis l'API Groq"""
        headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model,
            # 'input' must be a list of strings for embeddings endpoint
            "input": [text]
        }
        url = f"{self.base_url}/embeddings"
        print(f"[DEBUG] Embedding request URL: {url}")
        resp = self.session.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        # return the first embedding vector
        return resp.json()["data"][0]["embedding"]

    def chat_generate(self, messages, temperature=0.5):
        """Générer une réponse de chat depuis l'API Groq"""
        headers = {
            'Authorization': f"Bearer {self.api_key}",
            'Content-Type': 'application/json'
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        url = f"{self.base_url}/chat/completions"
        print(f"[DEBUG] Chat request URL: {url}")
        resp = self.session.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        class Response:
            pass

        r = Response()
        r.text = data["choices"][0]["message"]["content"]
        return r

    @property
    def chat(self):
        return self
