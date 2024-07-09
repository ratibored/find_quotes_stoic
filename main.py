from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
df = pd.read_pickle('embedded_citations.pkl')

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
