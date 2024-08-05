from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import uvicorn
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить запросы с любого источника
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST и т.д.)
    allow_headers=["*"],  # Разрешить все заголовки
)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
file_path = os.path.join(os.path.dirname(__file__), 'data', 'embedded_citations.pkl')
df = pd.read_pickle(file_path)

class QuoteRequest(BaseModel):
    quote: str

app.mount("/static", StaticFiles(directory="static"), name="static")

# Маршрут для отображения index.html из static
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


@app.post("/quotes/")
async def get_similar_quotes(request: QuoteRequest):
    user_citation = model.encode(getattr(request, 'quote'))
    df_ = df.copy()
    ans = []
    for stoic_citation in df_.embedded:
        result = round(util.pytorch_cos_sim(stoic_citation, user_citation).item()*100, 1)
        ans.append(result)
    df_['similarity'] = ans
    df_ = df_.sort_values('similarity', ascending=False).head(10).drop('embedded', axis=1)
    return {"quotes": df_.to_dict(orient='records')}


if __name__ == "__main__":
    uvicorn.run("app:app", host='localhost', port=8080, reload=True)
