FROM ollama/ollama:latest

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh

# Make the script executable
RUN chmod +x /entrypoint.sh

# Expose the required port
EXPOSE 11434

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
