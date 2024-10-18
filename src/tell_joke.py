import os
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic

from helpers import *

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

prompts = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
  ]

#use_gpt(model='gpt-3.5-turbo', openai=openai, temperature=0.7, prompts=prompts, stream=True)
#use_gpt(model='gpt-4o-mini', openai=openai, temperature=0.7, prompts=prompts, stream=False)
#use_gpt(model='gpt-4o', openai=openai, temperature=0.7, prompts=prompts, stream=True)
#use_claude(model='claude-3-5-sonnet-20240620', claude=claude, temperature=0.7, system_message=system_message, user_prompt=user_prompt, stream=True)
use_gemini(model='gemini-1.5-flash', system_message=system_message, user_prompt=user_prompt)