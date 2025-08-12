# 1️⃣ Use official lightweight Python image
FROM python:3.10-slim

# 2️⃣ Set environment variables for better Python behavior
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# 3️⃣ Set working directory
WORKDIR /app

# 4️⃣ Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5️⃣ Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6️⃣ Copy your entire project
COPY . .

# 7️⃣ Environment variables (Fly.io will override with secrets)
# These are placeholders, actual values will be set in Fly.io
# ENV DISCORD_TOKEN=changeme \
#     HF_API_KEY=changeme

# 8️⃣ Command to run the bot
CMD ["python", "discord_bot.py"]
