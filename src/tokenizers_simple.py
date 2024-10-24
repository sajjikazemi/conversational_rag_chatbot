import os
from huggingface_hub import login
from transformers import AutoTokenizer

HF_TOKEN = os.getenv('HF_TOKEN')
login(HF_TOKEN, add_to_git_credential=True)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B', trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
#print(tokens)
#len(tokens)
decoded = tokenizer.decode(tokens)
#print(decoded)
batch_decoded = tokenizer.batch_decode(tokens)
#print(batch_decoded)

vocab = tokenizer.vocab
#print(vocab)

added_vocab = tokenizer.get_added_vocab()
print(added_vocab)