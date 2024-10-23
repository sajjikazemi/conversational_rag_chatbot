import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import FluxPipeline, StableDiffusionPipeline

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')
login(HF_TOKEN, add_to_git_credential=True)

torch.cuda.empty_cache()

#pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
generator = torch.Generator(device="cuda").manual_seed(0)
prompt = "A futuristic class full of students learning AI coding"

pipe.enable_attention_slicing()

image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256,
    generator=generator
).images[0]

image.save("surreal.png")