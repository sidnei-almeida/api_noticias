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

# Carrega modelo e vetorizador
model_path = 'models/logreg.pkl'
vectorizer_path = 'models/vectorizer_logreg.pkl'
try:
    modelo = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"Erro ao carregar o modelo ou vetorizador: {e}")

# Feeds RSS
# Fontes RSS separadas por idioma
FEEDS_RSS_PT = [
    ("G1", "https://g1.globo.com/rss/g1/"),
    ("UOL", "https://rss.uol.com.br/feed/noticias.xml"),
    ("Estadão", "https://feeds.folha.uol.com.br/emcimadahora/rss091.xml"),
    ("Valor Econômico", "https://valor.globo.com/rss.xml"),
    ("El País", "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"),
    ("O Globo", "https://oglobo.globo.com/rss.xml"),
    ("R7", "https://noticias.r7.com/rss/"),
    ("Correio Braziliense", "https://www.correiobraziliense.com.br/rss/cidades.xml"),
    ("Agência Brasil", "https://agenciabrasil.ebc.com.br/rss/ultimasnews.xml"),
    ("Terra Notícias", "https://www.terra.com.br/rss/0,,EI1,00.xml"),
    ("IstoÉ", "https://istoe.com.br/feed/"),
    ("CartaCapital", "https://www.cartacapital.com.br/feed/"),
    ("Exame", "https://exame.com/feed/"),
]
FEEDS_RSS_EN = [
    ("BBC", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CNN", "http://rss.cnn.com/rss/edition.rss"),
    ("Reuters", "https://www.reuters.com/rssFeed/businessNews"),
    ("The Guardian", "https://www.theguardian.com/world/rss"),
    ("New York Times", "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("Associated Press", "https://apnews.com/rss/apf-topnews"),
    ("Al Jazeera English", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("CNBC", "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
    ("USA Today", "http://rssfeeds.usatoday.com/usatoday-NewsTopStories"),
    ("The Washington Post", "http://feeds.washingtonpost.com/rss/world"),
    ("ABC News", "https://abcnews.go.com/abcnews/topstories"),
    ("Politico", "https://www.politico.com/rss/politics08.xml"),
    ("Financial Times", "https://www.ft.com/?format=rss"),
    ("Bloomberg", "https://www.bloomberg.com/feed/podcast/etf-report.xml"),
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
    X_vec = vectorizer.transform([texto_limpo])
    pred = modelo.predict(X_vec)[0]
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
    X_vecs = vectorizer.transform([tl for _, tl in textos_validos])
    preds = modelo.predict(X_vecs)
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
    # Seleciona as fontes de acordo com o idioma
    if idioma == 'en':
        feeds = FEEDS_RSS_EN
    else:
        feeds = FEEDS_RSS_PT
    for fonte, url in feeds:
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
    X_vecs = vectorizer.transform([tl for _, tl in noticias_validas])
    preds = modelo.predict(X_vecs)
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

# Ponto de entrada para execução local
# No Render NÃO execute uvicorn.run manualmente, pois o Render já faz isso automaticamente.
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # 10000 é a porta padrão do Render
    # Para rodar localmente, acesse http://localhost:8000
    uvicorn.run("app.api:app", host="0.0.0.0", port=port)
