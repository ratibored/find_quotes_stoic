from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import torch
import os
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Указание полного пути к файлу embedded_citations.feather
file_path = os.path.join(os.path.dirname(__file__), 'embedded_citations.feather')

# Явно указываем устройство на CPU
device = torch.device('cpu')
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device=device) # cointegrated/rubert-tiny2 - слабее, но быстрее и легковеснее
df = pd.read_feather(file_path)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/quotes/", methods=["POST"])
def get_similar_quotes():
    request_data = request.get_json()
    user_citation = model.encode(request_data['quote'])
    df_ = df.copy()
    ans = []
    for stoic_citation in df_.embedded:
        result = round(util.pytorch_cos_sim(stoic_citation, user_citation).item() * 100, 1)
        ans.append(result)
    df_['similarity'] = ans
    df_ = df_.sort_values('similarity', ascending=False).head(10).drop('embedded', axis=1)
    return jsonify({"quotes": df_.to_dict(orient='records')})

if __name__ == "__main__":
    app.run(debug=True)