#!/usr/bin/env python3
"""
vk_crawler_full_with_followers.py
For each group listed in the JSON file, retrieves up to MAX_PER_GROUP of the most recent posts,
including: text, date, number of likes, comments, reposts, image URL, and the group's follower count.
Saves the results to a CSV file, skipping duplicates.

Group source: data/raw/vk_top_groups.json
"""

import time, itertools, json
from pathlib import Path
import pandas as pd
import vk_api, vk_api.exceptions as vk_exc
from my_config import ALL_VK_TOKENS as TOKENS

# ── Settings ───────────────────────────────────────────
JSON_PATH      = Path("data/raw/vk_top_groups.json")
CSV_PATH       = Path("data/raw/vk_top_groups_dataset_full.csv")
MAX_PER_GROUP  = 300
BATCH_SIZE     = 100
PAUSE          = 0.15

token_cycle = itertools.cycle(TOKENS)

def call(method, **params):
    while True:
        api = vk_api.VkApi(token=next(token_cycle), api_version="5.199").get_api()
        try:
            return getattr(api, method)(**params)
        except vk_api.exceptions.ApiError as e:
            if e.code in (6, 29):  # too many requests
                time.sleep(PAUSE)
                continue
            raise

def fetch_last_posts(group_id: int, max_n: int, known_post_ids: set):
    all_posts = []
    for offset in (0, BATCH_SIZE, 2 * BATCH_SIZE):
        if len(all_posts) >= max_n:
            break
        resp = call("wall.get", owner_id=-group_id, count=BATCH_SIZE, offset=offset, filter="owner")
        items = resp.get("items", [])
        for it in items:
            if it["id"] in known_post_ids:
                continue  # already saved
            txt = it.get("text", "").strip()
            if not txt:
                continue
            img = ""
            for att in it.get("attachments", []):
                if att["type"] == "photo":
                    img = att["photo"]["sizes"][-1]["url"]
                    break
            all_posts.append({
                "group_id":      group_id,
                "post_id":       it["id"],
                "from_id":       it.get("from_id"),
                "text":          txt,
                "date":          it["date"],
                "initial_date":  int(time.time()),
                "likes":         it.get("likes", {}).get("count", 0),
                "comments":      it.get("comments", {}).get("count", 0),
                "reposts":       it.get("reposts", {}).get("count", 0),
                "image_url":     img
            })
        time.sleep(PAUSE)
    return all_posts[:max_n]

def get_group_followers(group_id: int) -> int:
    try:
        resp = call("groups.getById", group_id=group_id, fields="members_count")
        if isinstance(resp, list) and "members_count" in resp[0]:
            return resp[0]["members_count"]
    except Exception as e:
        print(f"⚠️ Failed to get followers for group {group_id}: {e}")
    return 0

# ── Load group list ───────────────────────────────────────
with open(JSON_PATH, encoding="utf-8") as f:
    group_list = json.load(f)

# ── Load existing data if exists ──────────────────────────
if CSV_PATH.exists():
    existing_df = pd.read_csv(CSV_PATH)
    known_pairs = set(zip(existing_df["group_id"], existing_df["post_id"]))
    print(f"🔎 Found existing CSV with {len(existing_df)} rows")
else:
    existing_df = pd.DataFrame()
    known_pairs = set()

# ── Crawl groups ──────────────────────────────────────────
all_data = []
for group in group_list:
    vk_link = group["link"]
    group_id_str = vk_link.split("club")[-1]
    group_id = int(group_id_str)
    print(f"▶ Fetching posts from {group['name']} (ID: {group_id})")

    followers = get_group_followers(group_id)
    posts = fetch_last_posts(group_id, MAX_PER_GROUP, known_post_ids={pid for gid, pid in known_pairs if gid == group_id})

    if not posts:
        print("  ↪ Skipped (no new posts)")
        continue

    for post in posts:
        post["followers"] = followers
        post["source_name"] = group["name"]
        post["source_link"] = group["link"]
    all_data.extend(posts)

# ── Save merged result ────────────────────────────────────
if all_data:
    new_df = pd.DataFrame(all_data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)\
                    .drop_duplicates(subset=["group_id", "post_id"])
    combined_df.to_csv(CSV_PATH, index=False)
    print(f"✅ Appended {len(new_df)} → New total: {len(combined_df)}")
else:
    print("✅ No new posts — nothing to append.")

