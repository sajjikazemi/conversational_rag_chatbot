from website import Website
from openai import OpenAI

def user_prompt_for(website: Website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "The contents of this website is as follows; \
    please provide a short summary of this website in markdown. \
    If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

def messages_for(website: Website, system_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

def summarize(url: str, openai: OpenAI):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-40-mini",
        messages = messages_for(website) 
    )
    return response.choices[0].message.content