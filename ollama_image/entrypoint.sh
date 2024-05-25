#!/bin/sh

# Start the ollama service in the background
ollama serve &

# Wait for the service to start
sleep 10

# Pull the llama3 model
ollama pull llama3

# Bring ollama to the foreground
wait
