from website import Website
from openai import OpenAI
import json

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

def summarize(url: str, openai: OpenAI, system_prompt: str):
    website = Website(url)
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = messages_for(website, system_prompt) 
    )
    return response.choices[0].message.content

def get_links_user_prompt(website: Website):
        user_prompt = f"Here is the list of links on the website of {website.url} - "
        user_prompt += "please decide which of these are relevant web links for a brochure about the company, respond with the full https URL in JSON format. \
    Do not include Terms of Service, Privacy, email links.\n"
        user_prompt += "Links (some might be relative links):\n"
        user_prompt += "\n".join(website.links)
        return user_prompt

def get_links(url: str, openai: OpenAI, model, link_system_prompt: str):
    website = Website(url)
    completion = openai.chat.completions.create(
         model=model,
         messages=[
              {"role": "system", "content": link_system_prompt},
              {"role": "user", "content": get_links_user_prompt(website)}
         ],
         response_format={"type": "json_object"}
    )
    result = completion.choices[0].message.content
    return json.loads(result)