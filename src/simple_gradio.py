import os
from dotenv import load_dotenv
from openai import OpenAI
import anthropic
import google.generativeai
import gradio as gr

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')

openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure()

system_message = "You are helpful assistant"

def message_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    completion = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
    )
    return completion.choices[0].message.content

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

#gr.Interface(fn=shout, inputs="textbox", outputs="textbox").launch(share=True)

system_message = "You are a helpful assistant that responds in markdown"

view = gr.Interface(
    fn=message_gpt,
    inputs=[gr.Textbox(label="Your message:")],
    outputs=[gr.Markdown(label="Response:")],
    allow_flagging="never"
)
view.launch()