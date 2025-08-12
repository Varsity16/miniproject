import tkinter as tk
import customtkinter as ctk

from PIL import Image, ImageTk
from customtkinter import CTkImage

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler
from dotenv import load_dotenv
import os

load_dotenv()
auth_token = os.getenv("auth_token")

# from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

# # Load pipeline
# pipe = StableDiffusionPipeline.from_pretrained(
#     modelid,
#     revision="fp16",
#     torch_dtype=torch.float16,
#     use_auth_token=auth_token
# )
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.to(device)


# App window setup
app = tk.Tk()
app.geometry("532x700")
app.title("Stable Bud")

ctk.set_appearance_mode("dark")

# Prompt input field
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.pack(pady=20)

# Label to display generated image
lmain = ctk.CTkLabel(master=app, height=512, width=512, text="")
lmain.pack(pady=10)

# Load Stable Diffusion pipeline
# modelid = "CompVis/stable-diffusion-v1-4"
# modelid = "stabilityai/stable-diffusion-2-1"
# modelid = "stabilityai/sdxl-turbo"  #faster
modelid = "SG161222/Realistic_Vision_V5.1" #for realistic
# modelid = "reamlike-art/dreamlike-photoreal-2.0" #artistic
# modelid = "nitrosocke/Arcane-Diffusion" #cartoon


device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained(
    modelid,
    revision="fp16",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    use_auth_token=auth_token
)
pipe.to(device)

# Generate image function
def generate():
    with torch.autocast(device_type=device):
        prompt_text = prompt.get()
        if prompt_text.strip() == "":
            print("Please enter a prompt.")
            return

        #image = pipe(prompt_text, guidance_scale=12.0).images[0] #previously the guidance scale = 8.5
        # image = pipe(prompt_text, negative_prompt="blurry, low quality, deformed, bad anatomy", guidance_scale=12.0).images[0]
        # image = pipe(prompt_text, height=768, width=768,negative_prompt="blurry, low quality, deformed, bad anatomy", guidance_scale=12.0,num_inference_steps=70).images[0]
        # image = pipe(prompt, guidance_scale=15.0, num_inference_steps=75).images[0]
        
        
        image = pipe(
            prompt_text,
            guidance_scale=15.0,
            num_inference_steps=75
        ).images[0]


        
        # generator = torch.Generator(device).manual_seed(1234)
        # image = pipe(prompt_text, guidance_scale=12.0, generator=generator).images[0]




    image.save("generatedimage.png")
    img = ImageTk.PhotoImage(image)
    lmain.configure(image=img)
    lmain.image = img  # Keep reference to avoid garbage collection

# Generate button
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.pack(pady=10)

# Run the app
app.mainloop()
