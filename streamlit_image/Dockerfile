FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

# copy the rest of the application into the container at /app
COPY /local .

# make port 8501 available to the world outside this container
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]