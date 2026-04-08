FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose FastAPI (7860) and Gradio (7861)
EXPOSE 7860
EXPOSE 7861

CMD ["./start.sh"]
