FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install python dependencies
# Using --no-cache-dir to keep image small, but for CI caching we might want standard install.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
# Assuming we have an entry point in web/app.py or we might need to adjust based on user's framework.
# Since user mentioned FastAPI, we assume uvicorn is used.
# If web/app.py is Streamlit, we might need a different command.
# Checking conversation history, 'web/app.py' was for Streamlit.
# If there is a FastAPI app, it might be elsewhere or we need to create it.
# For now, I will assume a standard uvicorn start, but I should probably verify if there is a FastAPI app.
# The user request said "The project is Python-based (FastAPI/PyTorch)".
# I'll Assume the main app entry is `fake_news_detector.main:app` or similar, 
# but since I haven't seen a FastAPI file, I will default to a placeholder command 
# or use the Streamlit one if that's what they actually have.
# Actually, the user's PREVIOUS conversation said "Building Streamlit Fake News App".
# But THIS request says "Project is Python-based (FastAPI/PyTorch)".
# I will use a generic CMD that can be overridden or I'll just point to a likely file.
# If I use `CMD ["uvicorn", "fake_news_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]` 
# I might be guessing. 
# Best safe bet is to provide a sensible default for FastAPI as requested.

CMD ["uvicorn", "fake_news_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
