import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from embed import embed
from query import query
from get_vector_db import get_vector_db

load_dotenv()
TEMP_FOLDER = os.getenv('TEMP_FOLDER', './_temp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/')
def home():
    return "RAG API is running! Use /query or /embed endpoints."

@app.route('/embed', methods=['POST'])
def route_embed():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    embedded = embed(file)
    return jsonify({"message": "File embedded successfully"}) if embedded else jsonify({"error": "Embedding failed"}), 400

@app.route('/query', methods=['POST'])
def route_query():
    data = request.get_json()
    response = query(data.get('query'))
    return jsonify({"message": response}) if response else jsonify({"error": "Query failed"}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
