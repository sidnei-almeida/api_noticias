import pandas as pd
from app.preprocessing import limpar_dataframe, limpar_texto

# Função integrada para remover aspas da coluna texto
def remove_aspas_coluna_texto_df(df, coluna='texto'):
    if coluna in df.columns:
        df[coluna] = df[coluna].apply(lambda x: x[1:-1] if isinstance(x, str) and x.startswith('"') and x.endswith('"') else x)
    return df

# Caminho para o CSV de entrada e saída
csv_path = 'dados.csv'
csv_saida = 'dados.csv'

# 1. Ler os dados
print('Lendo dados de', csv_path)
df = pd.read_csv(csv_path)

# 2. Remover aspas da coluna texto
print('Removendo aspas da coluna texto...')
df = remove_aspas_coluna_texto_df(df, coluna='texto')

# 3. Aplicar o pré-processamento usando o preprocessing.py
print('Aplicando pré-processamento...')
df['texto'] = df['texto'].astype(str).apply(lambda x: limpar_texto(x, idioma='pt'))

# 4. Remover linhas vazias após limpeza
df = df[df['texto'].str.strip() != '']

# 5. Salvar o resultado final
print(f'Salvando dados limpos em {csv_saida}')
df.to_csv(csv_saida, index=False)
print(f'Dados limpos salvos em {csv_saida}')
