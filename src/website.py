import requests
from bs4 import BeautifulSoup

class Website:
    def __init__(self, url: str):
        self.url = url
        response = requests.get(url=url)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else 'No title found'
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]
    
    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpaga Contents:\n{self.text}\n\n"