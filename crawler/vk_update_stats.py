#!/usr/bin/env python3
"""
vk_update_stats.py
Updates likes, reposts, comments, media flags, and follower count
in data/raw/vk_top_groups_dataset_full.csv
"""

import pandas as pd
import time, itertools
from datetime import date
from pathlib import Path
import vk_api
from my_config import ALL_VK_TOKENS as TOKENS

# ── Configuration ─────────────────────────────────────────
CSV_PATH = Path("data/raw/vk_top_groups_dataset_full.csv")
PAUSE = 0.3
BATCH_SIZE = 100

# ── Setup ─────────────────────────────────────────────────
token_cycle = itertools.cycle(TOKENS)
today = date.today().isoformat()  # e.g. '2025-06-07'

def call(method, **params):
    """
    Call VK API with rotating tokens and basic rate-limit handling.
    """
    while True:
        api = vk_api.VkApi(token=next(token_cycle), api_version="5.199").get_api()
        try:
            return getattr(api, method)(**params)
        except vk_api.exceptions.ApiError as e:
            if e.code in (6, 29):  # too many requests
                print("⏳ Rate limit hit, sleeping...")
                time.sleep(PAUSE)
                continue
            raise

def get_followers_count(group_id):
    """
    Retrieve number of followers for a VK group.
    """
    try:
        resp = call("groups.getById", group_id=str(group_id), fields="members_count")
        if isinstance(resp, list) and len(resp) > 0:
            return resp[0].get('members_count', 0)
        if 'groups' in resp and len(resp['groups']) > 0:
            return resp['groups'][0].get('members_count', 0)
    except Exception as e:
        print(f"⚠️ Failed to get followers for group {group_id}: {e}")
    return 0

# ── Load CSV ──────────────────────────────────────────────
if not CSV_PATH.exists():
    raise FileNotFoundError(f"File not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"🔍 Loaded {len(df)} posts from {CSV_PATH}")

# ── Add new columns ───────────────────────────────────────
likes_col = f"likes_{today}"
reposts_col = f"reposts_{today}"
comments_col = f"comments_{today}"

df[likes_col] = None
df[reposts_col] = None
df[comments_col] = None

# Media and ad flags
for col in ["is_ad", "has_photo", "has_video", "has_link", "has_poll"]:
    if col not in df.columns:
        df[col] = None

# Follower column
if "followers" not in df.columns:
    df["followers"] = 0

# ── Update post stats via API ─────────────────────────────
for i in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[i:i + BATCH_SIZE]
    posts = [f"{row.from_id}_{row.post_id}" for _, row in batch.iterrows()]
    print(f"📦 Batch {i}: {len(posts)} posts")

    try:
        res = call("wall.getById", posts=posts)

        if i == 0:
            print("🔍 API response preview:", type(res), "\n", str(res)[:1000])

        if isinstance(res, dict):
            res = res.get("items", [])

        if not isinstance(res, list):
            print(f"🚫 Batch {i}: unexpected response type: {type(res)}. Skipping.")
            continue

        for idx, post_data in zip(batch.index, res):
            try:
                # Basic stats
                df.at[idx, likes_col] = post_data.get("likes", {}).get("count", 0)
                df.at[idx, reposts_col] = post_data.get("reposts", {}).get("count", 0)
                df.at[idx, comments_col] = post_data.get("comments", {}).get("count", 0)

                # Advertisement flag
                df.at[idx, "is_ad"] = int(post_data.get("marked_as_ads", 0) == 1)

                # Media flags
                attachments = post_data.get("attachments", [])
                df.at[idx, "has_photo"] = int(any(a.get("type") == "photo" for a in attachments))
                df.at[idx, "has_video"] = int(any(a.get("type") == "video" for a in attachments))
                df.at[idx, "has_link"]  = int(any(a.get("type") == "link" for a in attachments))
                df.at[idx, "has_poll"]  = int(any(a.get("type") == "poll" for a in attachments))

            except Exception as inner_e:
                print(f"⚠️ Failed to process post {idx}: {inner_e}")

    except Exception as e:
        print(f"🚫 Error in batch {i}: {e}")

    time.sleep(PAUSE)

# ── Update followers ──────────────────────────────────────
groups_to_update = df.loc[df['followers'] == 0, 'group_id'].dropna().unique()
print(f"🔁 Updating followers for {len(groups_to_update)} groups...")

followers_dict = {}
for gid in groups_to_update:
    followers_dict[gid] = get_followers_count(gid)
    print(f"👥 Group {gid}: {followers_dict[gid]} followers")
    time.sleep(PAUSE)

def update_followers(row):
    if row['followers'] == 0:
        return followers_dict.get(row['group_id'], 0)
    return row['followers']

df['followers'] = df.apply(update_followers, axis=1)

# ── Save CSV ──────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"✅ Updated {len(df)} rows. Columns added: {likes_col}, {reposts_col}, {comments_col}, media flags, and followers.")


