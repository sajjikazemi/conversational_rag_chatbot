import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

from website import Website
from helpers import *

# Load environment variables from the .env file
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
openai = OpenAI()

url = "https://cnn.com"
website = Website(url)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

summary = summarize(url, openai, system_prompt)
print(summary)