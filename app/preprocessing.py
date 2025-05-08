import re
import unicodedata
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Baixar stopwords se necessário
nltk.download('stopwords')

# Stopwords em português e inglês
stopwords_pt = set(stopwords.words('portuguese'))
stopwords_en = set(stopwords.words('english'))

# Termos não-noticiosos comuns (Português)
TERMOS_EXCLUIR = [
    'publicidade', 'continua depois da publicidade', 'leia mais', 'leia também', 'saiba mais',
    'conteúdo xp', 'análise do resultado', 'carteira', 'compra:', 'venda:', 'cotação', 'o que aconteceu',
    'confira a cotação', 'qual a cotação', 'dólar comercial', 'dólar turismo', 'data', 'link', 'idioma',
    'titulo', 'texto', 'fonte:', 'foto:', 'reportagem', 'assista', 'veja também', 'veja mais', 'vídeo',
    'imagem', 'divulgação', 'acesse', 'clique aqui', 'compartilhe', 'comentários', 'leitor', 'opinião',
    'newsletter', 'podcast', 'podcasts', 'podcast:', 'podcasts:', 'colunista', 'colunistas', 'editorial',
    'editoriais', 'redação', 'redacao', 'siga-nos', 'siga nos', 'facebook', 'twitter', 'instagram',
    'youtube', 'tiktok', 'patrocinado', 'conteúdo patrocinado', 'matéria exclusiva', 'matéria completa',
    'matéria', 'matérias', 'especial para o jornal', 'especial para o portal', 'reprodução', 'reproducao',
    'copyright', 'todos os direitos reservados', 'envie sua notícia', 'envie sua noticia', 'enviar notícia',
    'enviar noticia', 'últimas notícias', 'ultimas noticias', 'última atualização', 'ultima atualizacao',
    'atualizado em', 'fonte da notícia', 'fonte da noticia', 'fonte', 'foto', 'imagem', 'vídeo', 'video',
    'ver mais', 'ver detalhes', 'ver matéria', 'ver noticia', 'ver notícia', 'ver reportagem',
    'leia a seguir', 'leia reportagem', 'leia notícia', 'leia noticia', 'leia matéria', 'leia materia',
    'confira também', 'confira mais', 'confira detalhes', 'confira reportagem', 'confira notícia',
    'confira noticia', 'confira matéria', 'confira materia', 'saiba detalhes', 'saiba tudo', 'saiba antes',
    'veja detalhes', 'veja reportagem', 'veja notícia', 'veja noticia', 'veja matéria', 'veja materia',
    'veja tudo', 'veja aqui', 'veja agora', 'veja completa', 'veja completa',
]

# Termos não-noticiosos comuns (Inglês)
TERMOS_EXCLUIR_EN = [
    'advertisement', 'advertisements', 'sponsored', 'sponsored content', 'sponsored post',
    'read more', 'read also', 'see also', 'see more', 'click here', 'share', 'comments', 'opinion',
    'newsletter', 'podcast', 'podcasts', 'columnist', 'columnists', 'editorial', 'editorials',
    'newsroom', 'follow us', 'facebook', 'twitter', 'instagram', 'youtube', 'tiktok',
    'copyright', 'all rights reserved', 'exclusive report', 'full story', 'full article', 'article',
    'articles', 'special to the journal', 'special to the portal', 'reproduction',
    'send your news', 'send news', 'latest news', 'last update', 'updated on', 'news source',
    'source', 'photo', 'image', 'video', 'watch', 'see details', 'see article', 'see news',
    'see report', 'read next', 'read article', 'read news', 'read report', 'check also',
    'check more', 'check details', 'check report', 'check news', 'check article', 'learn more',
    'see details', 'see everything', 'see here', 'see now', 'see complete', 'see full',
]

# Função para remover acentuação

def remover_acentos(texto):
    return unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')

# Função aprimorada de limpeza de texto
def limpar_texto(texto, idioma='pt'):
    if not isinstance(texto, str):
        return ''
    texto = texto.lower().strip()
    texto = re.sub(r'\s+', ' ', texto)  # Normaliza espaços
    texto = re.sub(r"[^a-zà-úA-ZÀ-Ú\s]", "", texto)  # Remove caracteres especiais
    texto = texto.strip()
    # Remove termos não-noticiosos conforme idioma
    termos = TERMOS_EXCLUIR if idioma == 'pt' else TERMOS_EXCLUIR_EN
    for termo in termos:
        texto = texto.replace(termo, ' ')
    texto = re.sub(r'\s+', ' ', texto)
    # Remove acentuação para stopwords
    texto_sem_acentos = remover_acentos(texto)
    tokens = texto_sem_acentos.split()
    # Seleciona stopwords conforme idioma
    sw = stopwords_pt if idioma == 'pt' else stopwords_en
    tokens = [t for t in tokens if t not in sw and len(t) > 2]
    texto_final = " ".join(tokens)
    # Filtra textos muito curtos
    if len(texto_final.split()) < 5:
        return ''
    return texto_final

# Função para aplicar limpeza em DataFrame inteiro
def limpar_dataframe(df, coluna_texto='texto', coluna_idioma=None):
    """
    Limpa uma coluna de texto em um DataFrame.
    Se coluna_idioma for fornecida, usa o idioma de cada linha; senão, assume 'pt'.
    """
    def _limpa(row):
        lang = row[coluna_idioma] if coluna_idioma and coluna_idioma in row else 'pt'
        return limpar_texto(row[coluna_texto], lang)
    df["texto_limpo"] = df.apply(_limpa, axis=1)
    # Remove linhas vazias após limpeza
    df = df[df["texto_limpo"].str.strip() != '']
    return df