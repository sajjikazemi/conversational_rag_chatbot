from website import Website
from openai import OpenAI
import json
import anthropic
import google.generativeai

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

def get_all_details(url: str, openai: OpenAI, model, link_system_prompt):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url, openai=openai, model=model, link_system_prompt=link_system_prompt)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

def get_brochure_user_prompt(company_name: str, url: str, openai: OpenAI, model, link_system_prompt):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += get_all_details(url, openai=openai, model=model, link_system_prompt=link_system_prompt)
    user_prompt = user_prompt[:20_000] # Truncate if more than 20,000 characters
    return user_prompt

def create_brochure(company_name: str, url: str, openai: OpenAI, model, link_system_prompt, system_prompt: str):
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url, openai, model, link_system_prompt)}
        ],
    )
    result = response.choices[0].message.content
    return result

def use_gpt(model: str, openai: OpenAI, temperature: float, prompts, stream: bool = False):
    if not stream:
        completion = openai.chat.completions.create(
            model=model,
            messages=prompts,
            temperature=temperature
            )
        print(completion.choices[0].message.content)
    else:
        completion = openai.chat.completions.create(
            model=model,
            messages=prompts,
            temperature=temperature,
            stream=True
            )
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
        print()
        

def use_claude(model: str, claude: anthropic, temperature: float, system_message: str, user_prompt: str, stream: bool = False):
    if  not stream:    
        message = claude.messages.create(
            model=model,
            max_token=200,
            temperature=temperature,
            system=system_message,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        print(message.content[0].text)
    else:
        message = claude.messages.stream(
            model=model,
            max_tokens=200,
            temperature=temperature,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        with message as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)

def use_gemini(model: str, system_message: str, user_prompt: str):
    gemini = google.generativeai.GenerativeModel(
        model_name=model,
        system_instruction=system_message
    )
    response = gemini.generate_content(user_prompt)
    print(response.text)

def chat_with_openai(message: str, history: str, model: str, openai: OpenAI, system_message: str):
    messages = [{"role": "system", "content": system_message}]
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})
    if 'belt' in message:
        messages.append({"role": "system", "content": "For added context, the store does not sell belts, \
but be sure to point out other items on sale"})
    messages.append({"role": "user", "content": message})
    print("History is:")
    print(history)
    print("And messages is:")
    print(messages)
    stream = openai.chat.completions.create(model=model, messages=messages, stream=True)
    # response = ""
    # for chunk in stream:
    #     response += chunk.choices[0].delta.content or ''
    #     yield response
    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
    return response