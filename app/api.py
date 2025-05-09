import os
import joblib
import requests
import feedparser
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.preprocessing import limpar_texto  # Caminho absoluto funciona no Render

# Inicializa FastAPI
app = FastAPI(title="API de Classificação de Notícias")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo
model_path = 'models/model.pkl'
try:
    modelo = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo: {e}")

# Feeds RSS
FEEDS_RSS = [
    ("G1", "https://g1.globo.com/rss/g1/economia/"),
    ("Estadão", "https://economia.estadao.com.br/rss/ultimas.xml"),
    ("Valor Econômico", "https://valor.globo.com/rss.xml"),
    ("BBC", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CNN", "http://rss.cnn.com/rss/edition.rss"),
    ("El País", "https://elpais.com/rss/elpais/economia.xml"),
    ("Reuters", "https://www.reuters.com/rssFeed/businessNews"),
]

# Schemas
class NoticiaRequest(BaseModel):
    texto: str
    url: Optional[str] = None
    idioma: Optional[str] = None

class BatchRequest(BaseModel):
    textos: List[str]
    idioma: Optional[str] = None

class BuscarRequest(BaseModel):
    termo: str
    idioma: Optional[str] = None
    max_noticias: Optional[int] = 100

# Rotas
@app.get("/")
def root():
    return {"mensagem": "API de classificação de notícias ativa!"}

@app.post("/classificar/")
def classificar_noticia(req: NoticiaRequest):
    texto = req.texto
    if req.url:
        try:
            resp = requests.get(req.url, timeout=10)
            resp.raise_for_status()
            texto = resp.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao coletar URL: {e}")
    if not texto or len(texto.strip()) < 10:
        raise HTTPException(status_code=400, detail="Texto muito curto ou vazio.")
    idioma = req.idioma if req.idioma in ['pt', 'en'] else 'pt'
    texto_limpo = limpar_texto(texto, idioma=idioma)
    if not texto_limpo:
        raise HTTPException(status_code=400, detail="Texto ficou vazio após pré-processamento.")
    pred = modelo.predict([texto_limpo])[0]
    return {"texto_limpo": texto_limpo[:200] + ("..." if len(texto_limpo) > 200 else ""), "rotulo": pred}

@app.post("/classificar_batch/")
def classificar_batch(req: BatchRequest):
    if not req.textos or not isinstance(req.textos, list):
        raise HTTPException(status_code=400, detail="Envie uma lista de textos.")
    idioma = req.idioma if req.idioma in ['pt', 'en'] else 'pt'
    textos_limpos = [limpar_texto(t, idioma=idioma) for t in req.textos]
    textos_validos = [(t, tl) for t, tl in zip(req.textos, textos_limpos) if tl]
    if not textos_validos:
        raise HTTPException(status_code=400, detail="Nenhum texto válido após pré-processamento.")
    preds = modelo.predict([tl for _, tl in textos_validos])
    return {
        "resultados": [
            {"texto_limpo": tl[:100] + ("..." if len(tl) > 100 else ""), "rotulo": r}
            for (_, tl), r in zip(textos_validos, preds)
        ]
    }

@app.post("/buscar_classificar/")
def buscar_classificar(req: BuscarRequest):
    termo = req.termo.strip()
    if not termo or len(termo) < 2:
        raise HTTPException(status_code=400, detail="Informe um termo de busca válido.")
    idioma = req.idioma if req.idioma in ['pt', 'en'] else 'pt'
    max_noticias = req.max_noticias if req.max_noticias and req.max_noticias > 0 else 100
    noticias_encontradas = []
    for fonte, url in FEEDS_RSS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                titulo = entry.get('title', '')
                descricao = entry.get('description', '')
                link = entry.get('link', '')
                texto_busca = f"{titulo} {descricao}".lower()
                if termo.lower() in texto_busca:
                    noticias_encontradas.append({
                        'fonte': fonte,
                        'titulo': titulo,
                        'descricao': descricao,
                        'link': link
                    })
                if len(noticias_encontradas) >= max_noticias:
                    break
            if len(noticias_encontradas) >= max_noticias:
                break
        except Exception:
            continue
    if not noticias_encontradas:
        return {"resultados": [], "mensagem": "Nenhuma notícia encontrada para o termo informado."}
    textos_limpos = [limpar_texto(n['titulo'] + ' ' + n['descricao'], idioma=idioma) for n in noticias_encontradas]
    noticias_validas = [(n, tl) for n, tl in zip(noticias_encontradas, textos_limpos) if tl]
    if not noticias_validas:
        return {"resultados": [], "mensagem": "Nenhuma notícia válida após pré-processamento."}
    preds = modelo.predict([tl for _, tl in noticias_validas])
    resultados = []
    for (noticia, texto_limpo), rotulo in zip(noticias_validas, preds):
        resultados.append({
            "fonte": noticia['fonte'],
            "titulo": noticia['titulo'],
            "descricao": noticia['descricao'],
            "link": noticia['link'],
            "texto_limpo": texto_limpo[:200] + ("..." if len(texto_limpo) > 200 else ""),
            "rotulo": rotulo
        })
    return {"resultados": resultados, "total": len(resultados)}

# Ponto de entrada para execução local ou no Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)
