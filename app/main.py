"""FastAPI MVP: four-model comparison, selectable DistilBERT XAI, saved research metrics."""
from __future__ import annotations
import csv
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from app.explain import dispatch_explain

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 96
MODEL_PATHS = {
    'full': ROOT / 'models/distilbert/full_dataset_experiment/final_model',
    '10k': ROOT / 'models/distilbert/final_model',
}
BASELINE_PATHS = {
    'rf': ROOT / 'models/random_forest.joblib',
    'xgb': ROOT / 'models/xgboost.joblib',
    'tfidf': ROOT / 'models/tfidf_vectorizer.joblib',
}
# Full model on GPU if available; old model stays on CPU to avoid 4GB GPU OOM.
_models = {}
_tokenizers = {}
_baselines = {}
_load_errors = {}
_lock = threading.RLock()  # GPU XAI and concurrent predictions must not compete


def _load_distilbert(key):
    if key not in MODEL_PATHS:
        raise ValueError('Unknown DistilBERT model')
    if key not in _models:
        p = MODEL_PATHS[key]
        if not p.is_dir():
            raise FileNotFoundError(f'Model directory missing: {p}')
        tok = AutoTokenizer.from_pretrained(str(p), use_fast=True)
        target_device = DEVICE if key == 'full' else torch.device('cpu')
        mdl = AutoModelForSequenceClassification.from_pretrained(str(p)).to(target_device)
        mdl.eval()
        _tokenizers[key], _models[key] = tok, mdl
    return _models[key], _tokenizers[key], next(_models[key].parameters()).device


def _load_baseline(key):
    if key not in ('rf','xgb'):
        raise ValueError('Unknown baseline')
    import joblib
    if 'tfidf' not in _baselines:
        _baselines['tfidf'] = joblib.load(BASELINE_PATHS['tfidf'])
    if key not in _baselines:
        _baselines[key] = joblib.load(BASELINE_PATHS[key])
    return _baselines[key], _baselines['tfidf']


def _predict_one(url, key):
    t0 = time.perf_counter()
    if key in MODEL_PATHS:
        mdl,tok,device = _load_distilbert(key)
        enc=tok(url,return_tensors='pt',truncation=True,max_length=MAX_LENGTH)
        enc={k:v.to(device) for k,v in enc.items()}
        with torch.inference_mode():
            proba=mdl(**enc).logits.softmax(dim=-1)[0].detach().cpu().tolist()
    else:
        mdl,vectorizer=_load_baseline(key)
        # Match preprocessing in the original baseline training notebook.
        # Training preprocessing used lowercase and stripped URL strings.
        features=vectorizer.transform([url.strip().lower()])
        if not hasattr(mdl,'predict_proba'):
            raise ValueError(f'{key} model has no predict_proba method')
        raw=mdl.predict_proba(features)[0]
        classes=list(mdl.classes_)
        proba=[float(raw[classes.index(i)]) for i in (0,1)]
    cls=int(proba[1]>proba[0])
    return {'model_id':key,'model':{'rf':'Random Forest','xgb':'XGBoost','10k':'DistilBERT 10k','full':'DistilBERT Full'}[key],
            'prediction':'Malicious' if cls else 'Benign','predicted_class':cls,
            'confidence':round(float(proba[cls]),6),
            'probabilities':{'benign':round(float(proba[0]),6),'malicious':round(float(proba[1]),6)},
            'inference_time_ms':round((time.perf_counter()-t0)*1000,2)}


def _structure(url):
    parsed=urlsplit(url if '://' in url else 'http://'+url)
    return {'domain':parsed.netloc,'path':parsed.path,'query':parsed.query,
            'https':parsed.scheme.lower()=='https','length':len(url),
            'digit_count':sum(c.isdigit() for c in url),
            'special_character_count':sum(not c.isalnum() for c in url)}


@asynccontextmanager
async def lifespan(app):
    try:
        with _lock: _load_distilbert('full')
        print(f'Full-data DistilBERT loaded on {DEVICE}')
    except Exception as exc:
        _load_errors['full']=str(exc)
        print(f'WARNING: full model could not load: {exc}')
    yield

app=FastAPI(title='Explainable Malicious URL Detection',lifespan=lifespan)
app.mount('/static',StaticFiles(directory=str(ROOT/'app/static')),name='static')

class URLRequest(BaseModel):
    url:str=Field(min_length=1,max_length=4096)
    model:str='full'
    method:str='ig'


def _url(req):
    value=req.url.strip()
    if not value: raise HTTPException(400,'URL cannot be empty')
    return value

@app.get('/')
def home(): return FileResponse(str(ROOT/'app/templates/index.html'))

@app.get('/health')
def health():
    return {'status':'running','default_model':'full','device':str(DEVICE),
            'loaded_models':list(_models),'load_errors':_load_errors}

@app.post('/predict')
def predict(req:URLRequest):
    url=_url(req)
    if req.model not in MODEL_PATHS: raise HTTPException(400,'Select full or 10k')
    try:
        with _lock: result=_predict_one(url,req.model)
        return {'url':url,**result,'url_analysis':_structure(url),
                'warning':'Experimental prediction; not a verified website safety assessment.'}
    except Exception as exc:
        raise HTTPException(503,f'Model unavailable: {exc}') from exc

@app.post('/compare_models')
def compare_models(req:URLRequest):
    url=_url(req)
    output={}
    with _lock:
        for key in ('rf','xgb','10k','full'):
            try: output[key]=_predict_one(url,key)
            except Exception as exc: output[key]={'model_id':key,'error':str(exc)}
    return {'url':url,'default_model':'full','results':output,
            'warning':'Single-URL predictions are not held-out test performance metrics.'}

@app.post('/explain')
def explain(req:URLRequest):
    url=_url(req)
    if req.model not in MODEL_PATHS: raise HTTPException(400,'Select full or 10k')
    if req.method.lower() not in ('ig','lime','shap'):
        raise HTTPException(400,'Select ig, lime or shap')
    try:
        with _lock:
            mdl,tok,device=_load_distilbert(req.model)
            result=dispatch_explain(req.method,url,mdl,tok,device,max_length=MAX_LENGTH,n_steps=16)
        return {**result,'model_id':req.model,'url_analysis':_structure(url)}
    except ImportError as exc: raise HTTPException(503,str(exc)) from exc
    except Exception as exc: raise HTTPException(500,f'Explanation failed: {exc}') from exc


def _read_xai(path):
    if not path.exists(): return None
    result={}
    with path.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            method=row.get('method','').upper()
            if method not in ('IG','LIME','SHAP'): continue
            metrics={}
            for k in ('occlusion_spearman_mean','occlusion_spearman_std',
                      'deletion_probability_drop_mean','deletion_probability_drop_std',
                      'mean_cosine_similarity_mean','mean_cosine_similarity_std',
                      'explain_seconds_mean','explain_seconds_std',
                      'top_component_stability_mean','top_component_stability_std',
                      'prediction_flip_rate_mean','occlusion_spearman_count'):
                try: metrics[k]=round(float(row[k]),5)
                except (KeyError,TypeError,ValueError): metrics[k]=None
            result[method]=metrics
    return result or None

@app.get('/research_results')
def research_results():
    old=ROOT/'results/xai_200/summary_by_method.csv'
    new=ROOT/'results/xai_200_full_distilbert/summary_by_method.csv'
    metrics_file=ROOT/'results/distilbert_full_dataset/test_metrics.json'
    test=None
    if metrics_file.exists():
        try: test=json.loads(metrics_file.read_text(encoding='utf-8'))
        except (OSError,json.JSONDecodeError): pass
    return {'test_metrics':test,'xai':{'10k':_read_xai(old),'full':_read_xai(new)},
            'sources':{'10k':str(old.relative_to(ROOT)),'full':str(new.relative_to(ROOT))},
            'note':'XAI metrics are offline subset evaluations, not explanations of the current URL.'}
