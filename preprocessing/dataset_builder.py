# -*- coding: utf-8 -*-
"""
preprocessing/dataset_builder.py

Adds numeric/time/aux features on top of social_clean.* and fixes VK 2025-08-15 reposts.
Idempotent: if a target column already exists, we do NOT overwrite it unless noted.

Features:
  (1) Fix VK reposts_2025-08-15 → NaN
  (2) Time features: published_at_dt, hour_posted, dow_posted, is_weekend
  (3) Per-follower metrics: likes_pf, comments_pf, reposts_pf
  (4) Engagement change: engagement_change = er_final - engagement_rate
  (5) (Optional) TG repost deltas: reposts_delta_0607 / reposts_delta_0815 for platform=='tg'
  (6) Density features: emojis_per_100_chars, hashtags_per_100_chars, mentions_per_100_chars
  (7) len_text fallback, log_followers
  (8) TG forwarding flags: is_forwarded, has_fwd_src, has_fwd_msg_id
  (9) Viral label: is_viral = top 10% by engagement_rate per platform
  (10) (Optional) Sentiment if missing via ruBERT
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

EPS = 1e-9


def _to_dt_from_unix(s: pd.Series) -> pd.Series:
    s_num = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(s_num, unit="s", errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(
        {"1", "true", "t", "y", "yes", "да", "истина"}
    )


def build_dataset(
    df: pd.DataFrame,
    *,
    fix_vk_reposts_0815: bool = True,
    compute_time_features: bool = True,
    compute_pf_metrics: bool = True,
    compute_engagement_change: bool = True,
    compute_tg_reposts_deltas: bool = False,
    compute_density_features: bool = True,
    compute_viral_label: bool = True,
    enable_sentiment_if_missing: bool = False,
    sentiment_model_name: str = "blanchefort/rubert-base-cased-sentiment",
    sentiment_batch_size: int = 32,
) -> pd.DataFrame:
    out = df.copy()

    # ---------- (1) Fix VK reposts_2025-08-15 → NaN ----------
    # Description: For platform=='vk' rows, set reposts_2025-08-15 to NaN (it belongs to TG snapshot only).
    if fix_vk_reposts_0815 and {"platform", "reposts_2025-08-15"}.issubset(out.columns):
        mask_vk = out["platform"].astype(str).str.lower().isin({"vk", "вк", "vkontakte"})
        out.loc[mask_vk, "reposts_2025-08-15"] = np.nan

    # ---------- (2) Time features ----------
    # Description: Convert unix to datetime and derive hour/dow/weekend.
    if compute_time_features and "published_at" in out.columns:
        if "published_at_dt" not in out.columns:
            out["published_at_dt"] = _to_dt_from_unix(out["published_at"])
        if "hour_posted" not in out.columns:
            out["hour_posted"] = out["published_at_dt"].dt.hour
        if "dow_posted" not in out.columns:
            out["dow_posted"] = out["published_at_dt"].dt.dayofweek
        if "is_weekend" not in out.columns:
            out["is_weekend"] = out["dow_posted"].isin([5, 6]).astype(int)

    # ---------- (3) Per-follower metrics ----------
    # Description: Robust per-follower ratios for likes/comments/reposts if missing.
    if compute_pf_metrics and "followers" in out.columns:
        for raw_col, pf_col in [("likes", "likes_pf"), ("comments", "comments_pf"), ("reposts", "reposts_pf")]:
            if raw_col in out.columns and pf_col not in out.columns:
                out[pf_col] = out[raw_col].fillna(0) / (out["followers"].fillna(0) + EPS)

    # ---------- (4) Engagement change ----------
    # Description: engagement_change = er_final - engagement_rate (ONLY if both present).
    if compute_engagement_change and {"er_final", "engagement_rate"}.issubset(out.columns):
        out["engagement_change"] = out["er_final"] - out["engagement_rate"]

    # ---------- (5) TG repost deltas (optional) ----------
    # Description: For TG only, deltas to snapshots if both columns exist.
    if compute_tg_reposts_deltas and "platform" in out.columns and "reposts" in out.columns:
        is_tg = out["platform"].astype(str).str.lower().isin({"tg", "telegram", "тг", "телеграм"})
        if "reposts_2025-06-07" in out.columns and "reposts_delta_0607" not in out.columns:
            out.loc[is_tg, "reposts_delta_0607"] = out.loc[is_tg, "reposts"] - out.loc[is_tg, "reposts_2025-06-07"]
        if "reposts_2025-08-15" in out.columns and "reposts_delta_0815" not in out.columns:
            out.loc[is_tg, "reposts_delta_0815"] = out.loc[is_tg, "reposts"] - out.loc[is_tg, "reposts_2025-08-15"]

    # ---------- (6) Density features ----------
    # Description: Normalize counts by text length to reduce length bias.
    if compute_density_features:
        # len_text fallback
        if "len_text" not in out.columns:
            if "text_len" in out.columns:
                out["len_text"] = out["text_len"]
            elif "clean_text" in out.columns:
                out["len_text"] = out["clean_text"].fillna("").astype(str).str.len()
        denom = out.get("len_text", pd.Series(1, index=out.index)).replace(0, 1)

        if "num_emojis" in out.columns and "emojis_per_100_chars" not in out.columns:
            out["emojis_per_100_chars"] = 100.0 * out["num_emojis"].fillna(0) / denom
        if "num_hashtags" in out.columns and "hashtags_per_100_chars" not in out.columns:
            out["hashtags_per_100_chars"] = 100.0 * out["num_hashtags"].fillna(0) / denom
        if "num_mentions" in out.columns and "mentions_per_100_chars" not in out.columns:
            out["mentions_per_100_chars"] = 100.0 * out["num_mentions"].fillna(0) / denom

    # ---------- (7) log_followers ----------
    # Description: Stabilized follower scale for modeling.
    if "followers" in out.columns and "log_followers" not in out.columns:
        out["log_followers"] = np.log1p(out["followers"].astype(float))

    # ---------- (8) TG forwarding flags ----------
    # Description: Convenience booleans around TG forwarding info.
    if "tg_is_forwarded" in out.columns and "is_forwarded" not in out.columns:
        out["is_forwarded"] = _to_bool(out["tg_is_forwarded"])
    if "tg_fwd_src_id" in out.columns and "has_fwd_src" not in out.columns:
        out["has_fwd_src"] = out["tg_fwd_src_id"].notna().astype(int)
    if "tg_fwd_src_msg_id" in out.columns and "has_fwd_msg_id" not in out.columns:
        out["has_fwd_msg_id"] = out["tg_fwd_src_msg_id"].notna().astype(int)

    # ---------- (9) Viral label ----------
    # Description: is_viral = top 10% by engagement_rate per platform (fallback: global).
    if compute_viral_label and "engagement_rate" in out.columns:
        if "platform" in out.columns:
            thr = out.groupby(out["platform"].astype(str).str.lower())["engagement_rate"] \
                    .transform(lambda s: s.quantile(0.90))
        else:
            q90 = out["engagement_rate"].quantile(0.90)
            thr = pd.Series(q90, index=out.index)
        out["is_viral"] = (out["engagement_rate"] >= thr).astype(int)

    # ---------- (10) Sentiment if missing (optional) ----------
    # Description: If NO sentiment_* columns present, run ruBERT to infer them.
    if enable_sentiment_if_missing:
        have_sent = any(c in out.columns for c in
                        ["sentiment_label", "neg_prob", "neu_prob", "pos_prob", "sentiment_score"])
        if not have_sent:
            text_col = "clean_text" if "clean_text" in out.columns else ("text" if "text" in out.columns else None)
            if text_col is not None:
                try:
                    from transformers import AutoTokenizer, AutoModelForSequenceClassification
                    import torch

                    tok = AutoTokenizer.from_pretrained(sentiment_model_name)
                    mdl = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
                    mdl.eval()

                    labels_map = {0: "negative", 1: "neutral", 2: "positive"}
                    preds, neg_p, neu_p, pos_p = [], [], [], []
                    data = out[text_col].fillna("").astype(str).tolist()
                    for i in range(0, len(data), sentiment_batch_size):
                        batch = data[i:i+sentiment_batch_size]
                        enc = tok(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                        with torch.no_grad():
                            logits = mdl(**enc).logits
                            probs = torch.softmax(logits, dim=-1).cpu().numpy()
                            labs = probs.argmax(axis=1)
                        for j, lab in enumerate(labs):
                            preds.append(labels_map.get(int(lab), str(int(lab))))
                            neg_p.append(float(probs[j, 0]) if probs.shape[1] > 0 else 0.0)
                            neu_p.append(float(probs[j, 1]) if probs.shape[1] > 1 else 0.0)
                            pos_p.append(float(probs[j, 2]) if probs.shape[1] > 2 else 0.0)

                    out["sentiment_label"] = preds
                    out["neg_prob"] = neg_p
                    out["neu_prob"] = neu_p
                    out["pos_prob"] = pos_p
                    out["sentiment_score"] = out["pos_prob"].fillna(0) - out["neg_prob"].fillna(0)

                except Exception as e:
                    print(f"[dataset_builder] Sentiment inference skipped: {e}")

    return out


# ---------- file-level helper (optional in pipeline) ----------

def run_builder_file(
    in_path: Path,
    out_parquet: Optional[Path] = None,
    out_csv: Optional[Path] = None,
    **kwargs,
) -> pd.DataFrame:
    in_path = Path(in_path)
    if in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(in_path)

    df_final = build_dataset(df, **kwargs)

    if out_parquet:
        out_parquet = Path(out_parquet); out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(out_parquet, index=False)
    if out_csv:
        out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"✓ Rows: {len(df_final):,} | Cols: {df_final.shape[1]}")
    if out_parquet: print(f"  saved: {out_parquet}")
    if out_csv: print(f"  saved: {out_csv}")
    return df_final
