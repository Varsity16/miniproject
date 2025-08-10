import os
import discord
from discord.ext import commands
import requests
import io
from googletrans import Translator

# Load environment variables (Railway sets them automatically)
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face Model (Stable Diffusion XL Base)
HF_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Initialize Translator
translator = Translator()

# Discord Bot Setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

async def generate_image(prompt: str):
    """Call Hugging Face Inference API for image generation"""
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": prompt}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers, json=payload
    )
    
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        print(f"HF API Error: {response.status_code}, {response.text}")
        return None

@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}")

@bot.command()
async def imagine(ctx, *, prompt: str):
    await ctx.send("🌍 Translating and generating image...")
    
    translated_prompt = translator.translate(prompt, dest="en").text
    buffer = await generate_image(translated_prompt)

    if buffer:
        await ctx.send(file=discord.File(fp=buffer, filename="result.png"))
    else:
        await ctx.send("❌ Failed to generate image. Try again later.")

# Run the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
