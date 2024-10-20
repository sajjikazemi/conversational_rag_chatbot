import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from helpers import chat_with_openai

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

openai = OpenAI()
MODEL = 'gpt-4o-mini'

system_message = "You are a helpful assistant"

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales evemt.'\
Encourage the customer to buy hats if they are unsure what to get."

system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"

def chat(message, history):
    response_generator = chat_with_openai(message, history, MODEL, openai, system_message)
    full_response = ""
    for partial_response in response_generator:
        full_response += partial_response
    return full_response

gr.ChatInterface(fn=chat).launch(share=True)
