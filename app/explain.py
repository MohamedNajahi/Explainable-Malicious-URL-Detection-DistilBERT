"""Explainability methods (IG, LIME, SHAP) for fine-tuned DistilBERT URL classifier."""
from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlsplit

import torch
from captum.attr import LayerIntegratedGradients

# ---------------------------------------------------------------------------
# Optional LIME / SHAP imports
# ---------------------------------------------------------------------------
try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared URL regex (exact pattern from notebook)
# ---------------------------------------------------------------------------
URL_TOKEN_RE = re.compile(r'[A-Za-z0-9]+|[^A-Za-z0-9\s]')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def component_spans(url: str):
    """Character spans in the ORIGINAL URL (no text normalization for model input)."""
    prefix = 0 if '://' in url else len('http://')
    parsed = urlsplit(url if prefix == 0 else 'http://' + url)
    spans = {}
    # urlsplit yields indices for the normalized string; subtract synthetic prefix.
    normalized = url if prefix == 0 else 'http://' + url
    authority_start = normalized.find(parsed.netloc, normalized.find('://') + 3)
    if authority_start >= 0 and parsed.netloc:
        a, b = authority_start - prefix, authority_start - prefix + len(parsed.netloc)
        spans['domain'] = (max(0, a), min(len(url), b))
    if parsed.path:
        start = normalized.find(parsed.path, authority_start + len(parsed.netloc))
        if start >= 0:
            spans['path'] = (max(0, start - prefix), min(len(url), start - prefix + len(parsed.path)))
    if parsed.query:
        query_marker = normalized.find('?', authority_start + len(parsed.netloc))
        if query_marker >= 0:
            a = query_marker + 1 - prefix
            spans['query'] = (max(0, a), min(len(url), a + len(parsed.query)))
    return spans


def _map_tokens_to_components(url: str, token_list: list[dict], spans: dict) -> dict:
    """Accumulate attribution into URL components by character-overlap."""
    components = {key: 0.0 for key in ('domain', 'path', 'query')}
    for tok in token_list:
        a, b = tok['start'], tok['end']
        score = tok['attribution']
        for name, (c, d) in spans.items():
            overlap = max(0, min(b, d) - max(a, c))
            if overlap:
                components[name] += float(score) * overlap / (b - a)
    return components


# ---------------------------------------------------------------------------
# Integrated Gradients (original, UNCHANGED)
# ---------------------------------------------------------------------------

def explain_url(url: str, model, tokenizer, device, max_length: int = 96, n_steps: int = 16):
    if not getattr(tokenizer, 'is_fast', False):
        raise ValueError('Fast tokenizer required for character-offset explanations')
    enc = tokenizer(url, return_tensors='pt', truncation=True, max_length=max_length,
                    return_offsets_mapping=True)
    offsets = enc.pop('offset_mapping')[0].tolist()
    input_ids = enc['input_ids'].to(device)
    mask = enc['attention_mask'].to(device)
    baseline = torch.full_like(input_ids, tokenizer.pad_token_id)
    special = tokenizer.get_special_tokens_mask(input_ids[0].tolist(), already_has_special_tokens=True)
    for i, is_special in enumerate(special):
        if is_special:
            baseline[:, i] = input_ids[:, i]

    def forward(ids, attention_mask):
        return model(input_ids=ids, attention_mask=attention_mask).logits

    with torch.no_grad():
        logits = forward(input_ids, mask)
        probs = logits.softmax(dim=-1)[0]
        target = int(probs.argmax().item())

    lig = LayerIntegratedGradients(forward, model.get_input_embeddings())
    # Do not use torch.no_grad/inference_mode for IG: Captum requires gradients.
    attributions = lig.attribute(input_ids, baselines=baseline,
                                 additional_forward_args=(mask,), target=target,
                                 n_steps=n_steps, internal_batch_size=4)
    scores = attributions.sum(dim=-1)[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    spans = component_spans(url)
    components = {key: 0.0 for key in ('domain', 'path', 'query')}
    output_tokens = []
    for token, (a, b), score in zip(tokens, offsets, scores):
        if b <= a:
            continue
        output_tokens.append({'token': token, 'text': url[a:b], 'start': a, 'end': b,
                              'attribution': round(float(score), 6)})
        for name, (c, d) in spans.items():
            overlap = max(0, min(b, d) - max(a, c))
            if overlap:
                components[name] += float(score) * overlap / (b - a)
    return {
        'url': url,
        'prediction': 'Malicious' if target == 1 else 'Benign',
        'predicted_class': target,
        'confidence': round(float(probs[target].item()), 6),
        'probabilities': {'benign': round(float(probs[0].item()), 6),
                          'malicious': round(float(probs[1].item()), 6)},
        'tokens': output_tokens,
        'components': {k: round(v, 6) for k, v in components.items()},
        'method': 'Integrated Gradients (logit attribution)',
        'n_steps': n_steps,
        'warning': 'Experimental model; predictions are not verified website safety assessments.'
    }


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------

def explain_url_lime(url: str, model, tokenizer, device, max_length: int = 96,
                     lime_samples: int = 150):
    """LIME explanation using local linear surrogate with URL-aware tokenization."""
    if not LIME_AVAILABLE:
        raise ImportError('lime is not installed')

    # --- batch prediction closure ---
    def batch_proba(texts):
        enc = tokenizer(list(texts), return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = logits.softmax(dim=-1).cpu().numpy()
        return probs

    explainer = LimeTextExplainer(
        class_names=['benign', 'malicious'],
        split_expression=lambda text: URL_TOKEN_RE.findall(text),
        bow=False,
        random_state=42,
    )

    exp = explainer.explain_instance(
        url,
        batch_proba,
        num_features=len(URL_TOKEN_RE.findall(url)),
        num_samples=lime_samples,
        labels=[0, 1],
    )

    # --- get prediction from LIME's internal model call ---
    probs_single = batch_proba([url])[0]
    target = int(probs_single.argmax())

    # --- map LIME word positions back to character spans ---
    lime_tokens = URL_TOKEN_RE.findall(url)
    # Build char offsets for each LIME token via finditer
    token_char_spans = []
    for m in URL_TOKEN_RE.finditer(url):
        token_char_spans.append((m.group(), m.start(), m.end()))

    if len(token_char_spans) != len(lime_tokens):
        raise ValueError(
            f'LIME token count mismatch: findall={len(lime_tokens)}, '
            f'finditer={len(token_char_spans)}'
        )

    # LIME as_map() returns feature-index -> weight for the chosen class.
    # as_list() returns token text and may lose positional identity.
    weight_map = dict(exp.as_map().get(target, []))

    url_spans = component_spans(url)
    components = {key: 0.0 for key in ('domain', 'path', 'query')}
    output_tokens = []

    for idx, (tok_text, char_start, char_end) in enumerate(token_char_spans):
        weight = float(weight_map.get(idx, 0.0))
        output_tokens.append({
            'token': tok_text,
            'text': tok_text,
            'start': char_start,
            'end': char_end,
            'attribution': round(weight, 6),
        })
        a, b = char_start, char_end
        for name, (c, d) in url_spans.items():
            overlap = max(0, min(b, d) - max(a, c))
            if overlap:
                components[name] += weight * overlap / max(b - a, 1)

    return {
        'tokens': output_tokens,
        'components': {k: round(v, 6) for k, v in components.items()},
        'probabilities': {
            'benign': round(float(probs_single[0]), 6),
            'malicious': round(float(probs_single[1]), 6),
        },
        'predicted_class': target,
        'prediction': 'Malicious' if target == 1 else 'Benign',
        'confidence': round(float(probs_single[target]), 6),
    }


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def explain_url_shap(url: str, model, tokenizer, device, max_length: int = 96,
                     max_evals: int = 192):
    """SHAP partition explainer with text masker."""
    if not SHAP_AVAILABLE:
        raise ImportError('shap is not installed')

    import shap as _shap_local

    # --- batch prediction closure ---
    def batch_proba(texts):
        enc = tokenizer(list(texts), return_tensors='pt', padding=True,
                        truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = logits.softmax(dim=-1).cpu().numpy()
        return probs

    masker = _shap_local.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)
    explainer = _shap_local.Explainer(
        batch_proba,
        masker,
        algorithm='partition',
        output_names=['benign', 'malicious'],
    )

    shap_values = explainer([url], max_evals=max_evals)

    # shap_values.values shape: (1, n_tokens, 2)
    # shap_values.data[0] contains the tokenized strings SHAP used
    shap_tokens = list(shap_values.data[0])
    # Select scores for the predicted class (determined below).

    # --- get prediction ---
    probs_single = batch_proba([url])[0]
    target = int(probs_single.argmax())
    scores_malicious = shap_values.values[0, :, target].tolist()

    # --- align SHAP tokens to original URL characters ---
    cursor = 0
    output_tokens = []
    url_spans = component_spans(url)
    components = {key: 0.0 for key in ('domain', 'path', 'query')}

    for tok, score in zip(shap_tokens, scores_malicious):
        # Strip subword prefixes used by various tokenizers
        clean = tok.lstrip('##').lstrip('Ġ').lstrip('▁')
        if not clean:
            continue
        pos = url.find(clean, cursor)
        if pos == -1:
            raise ValueError(
                f'SHAP alignment failure: token "{tok}" (clean="{clean}") not found '
                f'in url[{cursor}:] = "{url[cursor:][:40]}"'
            )
        char_start = pos
        char_end = pos + len(clean)
        cursor = char_end

        output_tokens.append({
            'token': tok,
            'text': clean,
            'start': char_start,
            'end': char_end,
            'attribution': round(float(score), 6),
        })
        a, b = char_start, char_end
        for name, (c, d) in url_spans.items():
            overlap = max(0, min(b, d) - max(a, c))
            if overlap:
                components[name] += float(score) * overlap / max(b - a, 1)

    return {
        'tokens': output_tokens,
        'components': {k: round(v, 6) for k, v in components.items()},
        'probabilities': {
            'benign': round(float(probs_single[0]), 6),
            'malicious': round(float(probs_single[1]), 6),
        },
        'predicted_class': target,
        'prediction': 'Malicious' if target == 1 else 'Benign',
        'confidence': round(float(probs_single[target]), 6),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_explain(
    method: str,
    url: str,
    model,
    tokenizer,
    device,
    max_length: int = 96,
    n_steps: int = 16,
) -> dict:
    """Route explanation requests to the appropriate XAI method.

    Args:
        method: One of 'ig', 'lime', 'shap' (case-insensitive).
        url: URL string to explain.
        model: HuggingFace model.
        tokenizer: HuggingFace tokenizer.
        device: torch.device.
        max_length: Tokenizer max length.
        n_steps: IG integration steps.

    Returns:
        Dict with keys: url, prediction, predicted_class, confidence,
        probabilities, tokens, components, method, warning.
    """
    m = method.strip().lower()
    warning = 'Experimental model; predictions are not verified website safety assessments.'

    if m == 'ig':
        try:
            result = explain_url(url, model, tokenizer, device,
                                 max_length=max_length, n_steps=n_steps)
            result['warning'] = warning
            return result
        except Exception as exc:
            raise RuntimeError(f'IG explanation failed: {exc}') from exc

    elif m == 'lime':
        if not LIME_AVAILABLE:
            raise ImportError('LIME library (lime) is not installed. Install with: pip install lime')
        try:
            inner = explain_url_lime(url, model, tokenizer, device, max_length=max_length)
            return {
                'url': url,
                'prediction': inner['prediction'],
                'predicted_class': inner['predicted_class'],
                'confidence': inner['confidence'],
                'probabilities': inner['probabilities'],
                'tokens': inner['tokens'],
                'components': inner['components'],
                'method': 'LIME (Local linear approximation, 150 samples)',
                'warning': warning,
            }
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f'LIME explanation failed: {exc}') from exc

    elif m == 'shap':
        if not SHAP_AVAILABLE:
            raise ImportError('SHAP library (shap) is not installed. Install with: pip install shap')
        try:
            inner = explain_url_shap(url, model, tokenizer, device, max_length=max_length)
            return {
                'url': url,
                'prediction': inner['prediction'],
                'predicted_class': inner['predicted_class'],
                'confidence': inner['confidence'],
                'probabilities': inner['probabilities'],
                'tokens': inner['tokens'],
                'components': inner['components'],
                'method': 'SHAP (Shapley value partition explainer)',
                'warning': warning,
            }
        except ImportError:
            raise
        except Exception as exc:
            raise RuntimeError(f'SHAP explanation failed: {exc}') from exc

    else:
        raise ValueError(f'Unknown method: {method!r}. Must be one of: ig, lime, shap')
