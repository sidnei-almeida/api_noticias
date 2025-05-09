import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import optuna
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    lightgbm_installed = True
except ImportError:
    lightgbm_installed = False
    print('LightGBM não instalado. Só XGBoost será testado.')

def preprocess(text):
    return str(text).lower().strip()

# 1. Ler os dados limpos já pré-processados
df = pd.read_csv('noticias_limpo_sem_aspas_novo.csv')
df['texto'] = df['texto'].apply(preprocess)

# 2. Selecionar features e alvo
X = df['texto']
y = df['rotulo']

# 3. Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Vetorização
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Treinar e comparar modelos clássicos
modelos = {}
resultados = {}

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
acc_nb = accuracy_score(y_test, nb.predict(X_test_vec))
modelos['naive_bayes'] = nb
resultados['naive_bayes'] = acc_nb

# Logistic Regression
lr = LogisticRegression(max_iter=300, solver='lbfgs', class_weight='balanced', n_jobs=-1)
lr.fit(X_train_vec, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_vec))
modelos['logreg'] = lr
resultados['logreg'] = acc_lr

# SVM linear
svm = LinearSVC(class_weight='balanced', max_iter=1000)
svm.fit(X_train_vec, y_train)
acc_svm = accuracy_score(y_test, svm.predict(X_test_vec))
modelos['svm'] = svm
resultados['svm'] = acc_svm

# XGBoost
xgb_clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1)
xgb_clf.fit(X_train_vec, y_train)
acc_xgb = accuracy_score(y_test, xgb_clf.predict(X_test_vec))
modelos['xgboost'] = xgb_clf
resultados['xgboost'] = acc_xgb

# LightGBM
if lightgbm_installed:
    lgb_clf = lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1)
    lgb_clf.fit(X_train_vec, y_train)
    acc_lgb = accuracy_score(y_test, lgb_clf.predict(X_test_vec))
    modelos['lightgbm'] = lgb_clf
    resultados['lightgbm'] = acc_lgb

print('Acurácias dos modelos:')
for nome, acc in resultados.items():
    print(f"{nome}: {acc:.4f}")

# 6. Selecionar melhor modelo
melhor_nome = max(resultados, key=resultados.get)
print(f'\nMelhor modelo inicial: {melhor_nome} (acc={resultados[melhor_nome]:.4f})')

# 7. Otimização com Optuna para o melhor modelo

def objective_logreg(trial):
    C = trial.suggest_loguniform('C', 1e-3, 10.0)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])
    tol = trial.suggest_loguniform('tol', 1e-5, 1e-2)
    clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol, solver=solver, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return accuracy_score(y_test, y_pred)

def objective_svm(trial):
    C = trial.suggest_loguniform('C', 1e-3, 10.0)
    max_iter = trial.suggest_int('max_iter', 500, 3000)
    tol = trial.suggest_loguniform('tol', 1e-5, 1e-2)
    clf = LinearSVC(C=C, max_iter=max_iter, tol=tol, class_weight='balanced')
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return accuracy_score(y_test, y_pred)

def objective_xgb(trial):
    eta = trial.suggest_loguniform('eta', 1e-3, 0.3)
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    clf = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1,
                           eta=eta, n_estimators=n_estimators, max_depth=max_depth)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return accuracy_score(y_test, y_pred)

def objective_lgb(trial):
    num_leaves = trial.suggest_int('num_leaves', 15, 150)
    n_estimators = trial.suggest_int('n_estimators', 50, 400)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 0.3)
    clf = lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1,
                            num_leaves=num_leaves, n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    return accuracy_score(y_test, y_pred)

# Escolher função objetivo
if melhor_nome == 'logreg':
    objective = objective_logreg
    model_class = LogisticRegression
    model_file = 'models/logreg.pkl'
    vect_file = 'models/vectorizer_logreg.pkl'
elif melhor_nome == 'svm':
    objective = objective_svm
    model_class = LinearSVC
    model_file = 'models/svm.pkl'
    vect_file = 'models/vectorizer_svm.pkl'
elif melhor_nome == 'xgboost':
    objective = objective_xgb
    model_class = xgb.XGBClassifier
    model_file = 'models/xgb.pkl'
    vect_file = 'models/vectorizer_xgb.pkl'
elif melhor_nome == 'lightgbm' and lightgbm_installed:
    objective = objective_lgb
    model_class = lgb.LGBMClassifier
    model_file = 'models/lgb.pkl'
    vect_file = 'models/vectorizer_lgb.pkl'
else:
    print('Optuna tuning não implementado para o modelo:', melhor_nome)
    objective = None

# 8. Rodar Optuna para o melhor modelo
if objective is not None:
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=600, show_progress_bar=True)
    print('\nMelhores hiperparâmetros:', study.best_params)
    # Treinar modelo final com melhores hiperparâmetros
    if melhor_nome == 'logreg':
        final_model = LogisticRegression(**study.best_params, class_weight='balanced', n_jobs=-1)
    elif melhor_nome == 'svm':
        final_model = LinearSVC(**study.best_params, class_weight='balanced')
    elif melhor_nome == 'xgboost':
        final_model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, n_jobs=-1, **study.best_params)
    elif melhor_nome == 'lightgbm' and lightgbm_installed:
        final_model = lgb.LGBMClassifier(class_weight='balanced', n_jobs=-1, **study.best_params)
    else:
        final_model = None
    if final_model is not None:
        final_model.fit(X_train_vec, y_train)
        y_pred = final_model.predict(X_test_vec)
        print('\nRelatório final (teste):')
        print(classification_report(y_test, y_pred))
        print('Acurácia:', accuracy_score(y_test, y_pred))
        # Salvar modelo e vetorizador
        joblib.dump(final_model, model_file)
        joblib.dump(vectorizer, vect_file)
        print(f'Modelo e vetorizador salvos em {model_file} e {vect_file}')
else:
    print('Tuning Optuna não realizado para o modelo:', melhor_nome)

# Divisão treino/teste estratificada
treinamento, teste = train_test_split(df, test_size=0.2, random_state=42, stratify=df['rotulo'])

# Vetorização
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(treinamento['texto'])
X_test = vectorizer.transform(teste['texto'])
y_train = treinamento['rotulo']
y_test = teste['rotulo']

# 5. Modelos para testar
modelos = {
    'Naive Bayes': MultinomialNB(),
'Logistic Regression': LogisticRegression(max_iter=200, class_weight='balanced'),
'Linear SVM': LinearSVC(max_iter=2000, class_weight='balanced'),
# Exemplos adicionais:
# 'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
# 'Random Forest': RandomForestClassifier(class_weight='balanced', n_jobs=-1),
# 'Extra Trees': ExtraTreesClassifier(class_weight='balanced', n_jobs=-1),
'XGBoost': xgb.XGBClassifier(tree_method='hist', use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, scale_pos_weight=1),
}
if lightgbm_installed:
    modelos['LightGBM'] = lgb.LGBMClassifier(n_jobs=-1, class_weight='balanced')

# 6. Treinar e avaliar modelos
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for nome, modelo in modelos.items():
    print(f'\nTreinando e avaliando: {nome}')
    if nome in ['XGBoost', 'LightGBM']:
        # Codificar rótulos
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)
        modelo.fit(X_train, y_train_enc)
        y_pred_enc = modelo.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        y_true = y_test
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_true = y_test
    print('Acurácia:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))

# Dica: você pode salvar o melhor modelo com joblib/pickle se desejar
# Exemplo:
# import joblib
# joblib.dump(modelos['Linear SVM'], 'modelo_svm.joblib')
# joblib.dump(vectorizer, 'vectorizer.joblib')
