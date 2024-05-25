# Mongo DB  Atlas configuration information
username = 'team_user'
password = 'DPATMj5AeJ5EdFgp'

db_address = 'cluster0.orqic2o.mongodb.net'
db_name = 'medical_meadow_medqa'
collection_name = 'medical_data'
app_name = 'Cluster0'

mongo_uri = 'mongodb+srv://' + username + ':' + password + '@' + db_address + '/?retryWrites=true&w=majority&appName=' + app_name

# run Ollama locally on port 11434 - Docker based
ollama_url = 'http://localhost:11434'
# specify a model present on Ollama container
llm_model = 'llama3'

embedding_model = "sentence-transformers/paraphrase-MiniLM-L6-v2"
index_fields = "med_embedding"

