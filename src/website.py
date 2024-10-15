import requests
from bs4 import BeautifulSoup

class Website:
    def __init__(self, url: str):
        self.url = url
        response = requests.get(url=url)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else 'No title found'
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)