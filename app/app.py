# app.py
# ====== Dependencies (install once):
# pip install streamlit joblib numpy pandas xgboost sentence-transformers transformers emoji

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # force CPU
os.environ["TRANSFORMERS_NO_ACCELERATE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
import math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time
from joblib import load
import xgboost as xgb

# ---------- Models ----------
from sentence_transformers import SentenceTransformer
try:
    from transformers import pipeline
    _SENTIMENT_PIPE = pipeline(
        "sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
        tokenizer="blanchefort/rubert-base-cased-sentiment"
    )
except Exception:
    _SENTIMENT_PIPE = None


# ==========================
# Cache
# ==========================

@st.cache_resource
def load_xgb_model(path: str):
    """
    Returns: (model_like, feature_names or None)
    If booster has generic names f0,f1,... → feature_names=None (align by position).
    """
    obj = load(path)
    model = obj
    feature_names = None

    if isinstance(obj, dict):
        for key in ["model", "clf", "estimator", "xgb", "xgb_model", "best_estimator_"]:
            if key in obj:
                model = obj[key]
                break
        feature_names = obj.get("feature_names") or obj.get("features") or obj.get("columns")

    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and feature_names is None:
            names = getattr(booster, "feature_names", None)
            if names and not all(n.startswith("f") and n[1:].isdigit() for n in names):
                feature_names = names
            # else: treat as unnamed, align by position
    except Exception:
        pass

    return model, feature_names


@st.cache_resource
def load_sbert(name: str = "ai-forever/sbert_large_mt_nlu_ru"):
    # Must match training encoder. No fallback to avoid dim/name drift.
    return SentenceTransformer(name, device="cpu")


# ==========================
# Feature helpers
# ==========================

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE = re.compile(r"#[\w\-\_а-яА-ЯёЁ]+", re.UNICODE)
_MENTION_RE = re.compile(r"@[\w\-\_\.]+", re.UNICODE)
try:
    import emoji as _emoji
    _EMOJI_RE = _emoji.get_emoji_regexp()
except Exception:
    _EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")

def count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text))

def hour_sin_cos(hh: int, mm: int = 0) -> tuple[float, float]:
    frac = (hh + mm/60.0) / 24.0
    ang = 2 * math.pi * frac
    return math.sin(ang), math.cos(ang)

def sbert_embed(model: SentenceTransformer, text: str) -> np.ndarray:
    emb = model.encode([text], normalize_embeddings=False)
    return emb[0].astype("float32")

def sentiment_label(text: str) -> str:
    if _SENTIMENT_PIPE is not None:
        try:
            out = _SENTIMENT_PIPE(text[:512])
            lab = out[0]["label"].lower()
            if "neg" in lab: return "negative"
            if "pos" in lab: return "positive"
            return "neutral"
        except Exception:
            pass
    bad = len(re.findall(r"\b(плохо|ужас|страх|зло|тревога|ненавижу)\b", text.lower()))
    good = len(re.findall(r"\b(хорошо|класс|рад|люблю|спасибо|отлично)\b", text.lower()))
    if bad > good: return "negative"
    if good > bad: return "positive"
    return "neutral"

def build_meta_from_text(text: str, post_time: time,
                         likes: int, comments: int, followers: int,
                         has_photo: bool, has_video: bool) -> dict:
    length = len(text)
    num_hashtags = len(_HASHTAG_RE.findall(text))
    num_mentions = len(_MENTION_RE.findall(text))
    has_link = bool(_URL_RE.search(text))
    num_emj = count_emojis(text)
    emj_density = (num_emj / max(1, length)) * 100.0

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", text)
    uppers = [ch for ch in letters if ch.isupper()]
    caps_ratio = (len(uppers) / max(1, len(letters))) if letters else 0.0

    hsin, hcos = hour_sin_cos(post_time.hour, post_time.minute)
    likes_pf = (likes / followers) if followers > 0 else 0.0

    return {
        "len_text": float(length),
        "has_link": 1.0 if has_link else 0.0,
        "num_hashtags": float(num_hashtags),
        "num_mentions": float(num_mentions),
        "num_emojis": float(num_emj),
        "emojis_per_100_chars": float(emj_density),
        "caps_ratio": float(caps_ratio),
        "has_photo": 1.0 if has_photo else 0.0,
        "has_video": 1.0 if has_video else 0.0,
        "likes": float(likes),
        "comments": float(comments),
        "likes_pf": float(likes_pf),
        "hour_sin": float(hsin),
        "hour_cos": float(hcos),
        "platform_vk": 1.0,
        "platform_tg": 0.0,
    }

def _required_dim_from_feature_names(feature_names: list[str], prefix: str) -> int:
    idxs = []
    for c in feature_names or []:
        if c.startswith(prefix):
            try:
                idxs.append(int(c.split("_")[1]))
            except Exception:
                pass
    return (max(idxs) + 1) if idxs else 0

def _coerce_dim(vec: np.ndarray, needed: int) -> np.ndarray:
    d = vec.shape[0]
    if needed == 0 or d == needed: return vec
    if d > needed: return vec[:needed]
    out = np.zeros((needed,), dtype=vec.dtype)
    out[:d] = vec
    return out

def make_feature_row(expected_cols: list[str] | None,
                     sbert_vec: np.ndarray,
                     meta: dict) -> pd.DataFrame:
    features = {}

    # Conform SBERT dim to model expectations if names are provided
    needed_sbert = _required_dim_from_feature_names(expected_cols or [], "sbert_")
    if needed_sbert > 0:
        sbert_vec = _coerce_dim(sbert_vec, needed_sbert)

    # 1) SBERT
    for i, v in enumerate(sbert_vec):
        features[f"sbert_{i}"] = float(v)

    # 2) META
    for k, v in meta.items():
        features[k] = float(v)

    # 3) If model expects clip_emb_*, make them present (zeros if no image)
    if expected_cols:
        for c in (c for c in expected_cols if c.startswith("clip_emb_")):
            features[c] = features.get(c, 0.0)

    # 4) Build DataFrame with strict order if expected_cols provided
    if expected_cols:
        row = {c: features.get(c, 0.0) for c in expected_cols}
        X = pd.DataFrame([row], columns=expected_cols)
    else:
        cols = sorted(features.keys())
        X = pd.DataFrame([{c: features.get(c, 0.0) for c in cols}], columns=cols)
    return X


# ==========================
# Robust prediction via Booster
# ==========================

def _booster(model):
    if isinstance(model, xgb.Booster):
        return model
    if hasattr(model, "get_booster"):
        return model.get_booster()
    if isinstance(model, dict) and isinstance(model.get("booster"), xgb.Booster):
        return model["booster"]
    return None

def _is_f_names(names):
    return bool(names) and all(n.startswith("f") and n[1:].isdigit() for n in names)

def _align_to_expected(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra)
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    return X[expected]

def predict_xgb_any(model, X: pd.DataFrame) -> float:
    """
    Returns probability for class 1.
    Always routes through raw Booster to control feature name handling.
    """
    bst = _booster(model)
    if bst is None:
        # Fallback: use sklearn path
        if hasattr(model, "predict_proba"):
            proba = float(np.asarray(model.predict_proba(X))[:, 1][0])
            return proba
        if hasattr(model, "decision_function"):
            margin = float(np.asarray(model.decision_function(X)).ravel()[0])
            return 1.0 / (1.0 + np.exp(-margin))
        y = float(np.asarray(model.predict(X)).ravel()[0])
        return y if 0.0 <= y <= 1.0 else (1.0 if y > 0 else 0.0)

    names = bst.feature_names  # None | ['f0',...] | explicit names
    if _is_f_names(names) or not names:
        dmat = xgb.DMatrix(X.values)  # align by position
    else:
        X = _align_to_expected(X.copy(), names)
        dmat = xgb.DMatrix(X, feature_names=names)

    p = float(bst.predict(dmat).ravel()[0])
    return p if 0.0 <= p <= 1.0 else 1.0 / (1.0 + np.exp(-p))


# ==========================
# UI
# ==========================

st.set_page_config(page_title="VK Post Virality Predictor", page_icon="🔥", layout="centered")
st.title("VK Post Virality Predictor (Fusion XGBoost)")

st.write(
    "Paste the post text below. The app will build an SBERT embedding + simple metadata "
    "and estimate the probability of virality. Sentiment is displayed for convenience and "
    "**does not** affect the XGBoost prediction."
)

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value="../models/fusion_xgb.joblib")
    thr = st.slider("Decision threshold (τ)", 0.0, 1.0, 0.255, 0.005,
                    help="If p ≥ τ → 'Viral', else 'Not viral'.")
    st.caption(
        "Provide basic metadata (at publication time these are typically ≈0, "
        "but you can set them if you test a post that already gathered reactions)."
    )
    post_time = st.time_input(
        "Planned publish time",
        value=datetime.now().time().replace(second=0, microsecond=0)
    )
    likes = st.number_input("Likes (at evaluation time)", min_value=0, value=0, step=1)
    comments = st.number_input("Comments (at evaluation time)", min_value=0, value=0, step=1)
    followers = st.number_input("Followers (for likes_pf)", min_value=0, value=0, step=100)
    has_photo = st.checkbox("has_photo", value=False)
    has_video = st.checkbox("has_video", value=False)

user_text = st.text_area("VK/TG post text:", height=220, placeholder="Paste the post text...")

colA, colB = st.columns(2)
with colA:
    do_predict = st.button("Predict")
with colB:
    st.write("")

if do_predict:
    if not user_text.strip():
        st.error("Please enter text for analysis.")
        st.stop()

    with st.spinner("Loading model and embedder..."):
        model, feature_names = load_xgb_model(model_path)
        sbert = load_sbert()
        bst = model if isinstance(model, xgb.Booster) else getattr(model, "get_booster", lambda: None)()
        booster_feature_names = None
        if bst is not None and getattr(bst, "feature_names", None):
            names = bst.feature_names
            if not (names and all(n.startswith("f") and n[1:].isdigit() for n in names)):
                booster_feature_names = names
        expected_cols = feature_names or booster_feature_names  # может быть None

    meta = build_meta_from_text(
        text=user_text,
        post_time=post_time,
        likes=likes,
        comments=comments,
        followers=followers,
        has_photo=has_photo,
        has_video=has_video,
    )
    sbert_vec = sbert_embed(sbert, user_text)
    X = make_feature_row(expected_cols, sbert_vec, meta)

    # если у бустера есть явные имена, выравниваем строго под них
    if bst is not None and bst.feature_names and not _is_f_names(bst.feature_names):
        X = _align_to_expected(X, bst.feature_names)

    proba = predict_xgb_any(model, X)
    is_viral = proba >= thr

    sent = sentiment_label(user_text)

    st.subheader("Result")
    if is_viral:
        st.success(f"This post is LIKELY VIRAL (p = {proba:.3f}, τ = {thr:.3f})")
    else:
        st.warning(f"This post is LIKELY NOT VIRAL (p = {proba:.3f}, τ = {thr:.3f})")

    st.write(f"**Sentiment (UI):** {sent.capitalize()}")

