# Use official Python 3.10 (or 3.13) image
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy only what we need first (leverage Docker cache)
COPY requirements.txt .

# Install system deps (if any) and Python deps
RUN apt-get update && \
    apt-get install -y build-essential curl git && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your code
COPY . .

# Tell Streamlit to listen on 0.0.0.0:  
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0  
ENV STREAMLIT_SERVER_PORT=8080

# Expose port  
EXPOSE 8080

# Command to run your app
CMD ["streamlit", "run", "agent.py"]
