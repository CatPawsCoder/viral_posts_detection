# -*- coding: utf-8 -*-
"""
preprocessing/text_cleaner.py

Text cleaning and feature extraction:
• extract emojis into a separate column (emojis)
• replace emojis with Russian words (extended dictionary)
• remove links / html / hashtags / @mentions, normalization
• lemmatization (pymorphy3) + stopword removal
• counters: num_emojis / num_hashtags / num_mentions
• ad detector is_ad_by_text and final ad flag is_ad_final

Usage in pipeline:
    from preprocessing.text_cleaner import process_text_features
    df = process_text_features(df)
"""

from __future__ import annotations
import re, html, ssl
import pandas as pd
from pymorphy3 import MorphAnalyzer

# ── emoji (optional, only if package installed) ───────────────
try:
    import emoji
    HAVE_EMOJI = True
except Exception:
    HAVE_EMOJI = False

# ── stopwords (nltk) ─────────────────────────────────────────
try:
    from nltk.corpus import stopwords
    stopru = set(stopwords.words("russian"))
except Exception:
    import nltk
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stopru = set(stopwords.words("russian"))

morph = MorphAnalyzer()

# ── regex patterns ───────────────────────────────────────────
HTML_TAGS  = re.compile(r"<[^>]+>")
URL_RE     = re.compile(r"https?://\S+|vk\.cc/\S+", re.IGNORECASE)
TME_RE     = re.compile(r"(?:https?://)?t\.me/[^\s]+", re.IGNORECASE)
HASHTAG    = re.compile(r"#(\w+)")
AT_MENT    = re.compile(r"(?<!\w)@([A-Za-z0-9_]+)")
NON_CYR    = re.compile(r"[^а-яё0-9 .,!?;:()\n\-]+", re.IGNORECASE)
MULTI_SP   = re.compile(r"\s+")

# ── emoji → Russian words (extended dictionary) ──────────────
EMOJI_RU = {
    "😂": "смех", "🤣": "гомерический смех", "😊": "улыбка", "😍": "влюблённость",
    "😒": "раздражение", "😘": "поцелуй", "😁": "радость", "😭": "плач",
    "😩": "усталость", "😔": "грусть", "😏": "самодовольство", "😎": "крутость",
    "😢": "печаль", "😡": "злость", "😇": "невинность", "😅": "облегчение",
    "🙄": "скепсис", "😤": "раздражение", "😱": "испуг", "🤔": "размышление",
    "👍": "лайк", "👎": "дизлайк", "👌": "окей", "✌️": "мир", "🤞": "надежда",
    "🙏": "молитва", "👏": "аплодисменты", "🙌": "радость", "💪": "сила",
    "🧠": "мозг", "🫶": "сердце руками", "❤️": "любовь", "🧡": "оранжевое сердце",
    "💛": "жёлтое сердце", "💚": "зелёное сердце", "💙": "синее сердце",
    "💜": "фиолетовое сердце", "🖤": "чёрное сердце", "🤍": "белое сердце",
    "🤎": "коричневое сердце", "💔": "разбитое сердце", "🔥": "огонь", "✨": "сияние",
    "🌟": "звезда", "🌈": "радуга", "🎉": "праздник", "🎊": "конфетти", "🎁": "подарок",
    "🎂": "торт", "🍰": "десерт", "🍕": "пицца", "🍔": "бургер", "🍟": "картошка",
    "🍗": "курочка", "🍺": "пиво", "🍷": "вино", "🥂": "тост", "☕": "кофе", "🍵": "чай",
    "🍼": "бутылочка", "🚗": "машина", "🚀": "ракета", "✈️": "самолёт", "🚁": "вертолёт",
    "🏠": "дом", "🏢": "здание", "📱": "смартфон", "💻": "ноутбук", "📷": "камера",
    "🎧": "наушники", "🎮": "игра", "🎵": "музыка", "📚": "книги", "📖": "чтение",
    "📝": "запись", "💡": "идея", "🔒": "замок", "🔓": "открытый замок", "🛒": "покупка",
    "💸": "деньги", "💰": "сумка с деньгами", "🪙": "монета", "🏆": "трофей",
    "🥇": "золото", "🥈": "серебро", "🥉": "бронза", "👑": "корона", "⚽": "футбол",
    "🏀": "баскетбол", "🏈": "регби", "⚾": "бейсбол", "🥊": "бокс", "🏓": "пинг-понг",
    "⛷️": "лыжи", "🏋️": "тяжёлая атлетика", "🚴": "велосипед"
}

# extract distinct emojis preserving order
def _distinct_emoji_list(s: str) -> list[str]:
    if not HAVE_EMOJI or not isinstance(s, str):
        return []
    return emoji.distinct_emoji_list(s)

# return emojis as a space-separated string
def extract_emojis(text: str) -> str:
    return " ".join(_distinct_emoji_list(text))

# replace emojis with Russian words
def replace_emojis_with_words(text: str) -> str:
    if not HAVE_EMOJI or not isinstance(text, str):
        return str(text)
    t = text
    for emo in _distinct_emoji_list(text):
        word = EMOJI_RU.get(emo, "")
        t = t.replace(emo, f" {word} " if word else " ")
    return t

# ── ad detection patterns ─────────────────────────────────────
ADS_PATTERNS = [
    r"\bна\s*правах\s*реклам[ыы]?\b",
    r"\bреклам[ауеы]?\b", r"\bспонсор\w*\b", r"\bпартн[её]рск\w*\b",
    r"\bпромо\-?код\w*\b|\bпромокод\w*\b", r"\bскидк\w*\b",
    r"\bкупи(ть|)\b|\bпокупай\b|\bзакажи(те|)\b|\bоформить\s*заказ\b",
    r"\bподпис(ывайтесь|ка|аться|ывайся)\b",
    r"\bрозыгрыш\b|\bконкурс\b|\bgiveaway\b",
    r"\b-?\d{1,3}\s?%(\s*скидк\w*)?\b",
    r"\b\d{3,}\s?(₽|руб\.?|р\.?|сом|uzs|usd|\$|€)\b",
    r"\b\+?\d[\d\s\-\(\)]{7,}\b",
]
ADS_RE = re.compile("|".join(ADS_PATTERNS), re.IGNORECASE | re.UNICODE)

def detect_ads(text: str) -> bool:
    return bool(text) and ADS_RE.search(str(text)) is not None

# ── normalization and lemmatization ──────────────────────────
def normalize_text(text: str) -> str:
    t = html.unescape(str(text)).lower()
    t = TME_RE.sub(" ", t)        # t.me links
    t = URL_RE.sub(" ", t)        # http/https/vk.cc links
    t = HTML_TAGS.sub(" ", t)     # HTML tags
    t = HASHTAG.sub(r"\1", t)     # remove #
    t = AT_MENT.sub(r"\1", t)     # remove @
    t = replace_emojis_with_words(t)
    t = NON_CYR.sub(" ", t)       # non-Cyrillic chars
    t = MULTI_SP.sub(" ", t).strip()
    return t

def lemmatize(text: str) -> str:
    return " ".join(
        morph.parse(w)[0].normal_form
        for w in text.split()
        if w and w not in stopru
    )

# ── main processing pipeline ─────────────────────────────────
def process_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns to dataframe:
      emojis, num_emojis, num_hashtags, num_mentions,
      is_ad_by_text, clean_text, lemma_text, is_ad_final
    Returns a new dataframe (does not save to disk).
    """
    out = df.copy()
    out["text"] = out.get("text", "").fillna("").astype(str)

    # 1) extract emojis and counts before normalization
    if HAVE_EMOJI:
        out["emojis"] = out["text"].apply(extract_emojis)
        out["num_emojis"] = out["text"].apply(lambda s: len(_distinct_emoji_list(s)))
    else:
        out["emojis"] = ""
        out["num_emojis"] = 0

    # 2) count hashtags and mentions
    out["num_hashtags"] = out["text"].str.count(r"#\w+", flags=re.IGNORECASE).fillna(0).astype(int)
    out["num_mentions"] = out["text"].str.count(r"(?<!\w)@\w+").fillna(0).astype(int)

    # 3) detect ads by text
    out["is_ad_by_text"] = out["text"].apply(detect_ads)

    # 4) normalize and lemmatize
    out["clean_text"] = out["text"].apply(normalize_text)
    out["lemma_text"] = out["clean_text"].apply(lemmatize)

    # 5) final ad flag: original is_ad OR detected ads
    if "is_ad" in out.columns:
        base_ad = out["is_ad"].astype(str).str.strip().str.lower().isin({"1","true","t","да","y"})
    else:
        base_ad = False
    out["is_ad_final"] = (pd.Series(base_ad, index=out.index) | out["is_ad_by_text"]).astype(int)

    # 6) clean corpus: remove very short texts and duplicates
    out = out[out["lemma_text"].str.len() > 5].drop_duplicates(subset=["lemma_text"]).copy()
    return out




# #!/usr/bin/env python3
# """
# preprocessing/cleaner.py
# ────────────────────────────────────────
# Создаёт единый data/preprocessed/vk_clean.csv из raw CSV.

# • Распознаёт рекламу по ключевым признакам
# • Извлекает и сохраняет эмодзи (в emojis)
# • Заменяет эмодзи на текст (на русском)
# • Приводит к нижнему регистру, убирает html, ссылки, хештеги и @
# • Удаляет стоп-слова и делает лемматизацию

# Запуск:
#     python -m preprocessing.cleaner
# """

# import re, html, sys, emoji, ssl
# from pathlib import Path
# import pandas as pd
# from pymorphy3 import MorphAnalyzer

# # ── стоп-слова ────────────────────────────────
# try:
#     from nltk.corpus import stopwords
#     stopru = set(stopwords.words("russian"))
# except Exception:
#     import nltk
#     ssl._create_default_https_context = ssl._create_unverified_context
#     nltk.download("stopwords")
#     from nltk.corpus import stopwords
#     stopru = set(stopwords.words("russian"))

# # ── пути ──────────────────────────────────────
# RAW_CSV  = Path("data/raw/vk_top_groups_dataset_full.csv")
# OUT_CSV  = Path("data/preprocessed/vk_clean.csv")
# OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# # ── regex ─────────────────────────────────────
# URL_RE     = re.compile(r"https?://\S+|vk\.cc/\S+")
# HTML_TAGS  = re.compile(r"<[^>]+>")
# HASHTAG    = re.compile(r"#(\w+)")
# MENTION    = re.compile(r"@(\w+)")
# NON_CYR    = re.compile(r"[^а-яё0-9 .,!?;:()\n\-]+", re.I)
# MULTI_SP   = re.compile(r"\s+")

# # ── морфология ────────────────────────────────
# morph = MorphAnalyzer()

# EMOJI_RU = {
#     "😂": "смех", "🤣": "гомерический смех", "😊": "улыбка", "😍": "влюблённость",
#     "😒": "раздражение", "😘": "поцелуй", "😁": "радость", "😭": "плач",
#     "😩": "усталость", "😔": "грусть", "😏": "самодовольство", "😎": "крутость",
#     "😢": "печаль", "😡": "злость", "😇": "невинность", "😅": "облегчение",
#     "🙄": "скепсис", "😤": "раздражение", "😱": "испуг", "🤔": "размышление",
#     "👍": "лайк", "👎": "дизлайк", "👌": "окей", "✌️": "мир", "🤞": "надежда",
#     "🙏": "молитва", "👏": "аплодисменты", "🙌": "радость", "💪": "сила",
#     "🧠": "мозг", "🫶": "сердце руками", "❤️": "любовь", "🧡": "оранжевое сердце",
#     "💛": "жёлтое сердце", "💚": "зелёное сердце", "💙": "синее сердце", "💜": "фиолетовое сердце",
#     "🖤": "чёрное сердце", "🤍": "белое сердце", "🤎": "коричневое сердце", "💔": "разбитое сердце",
#     "🔥": "огонь", "✨": "сияние", "🌟": "звезда", "🌈": "радуга", "🎉": "праздник",
#     "🎊": "конфетти", "🎁": "подарок", "🎂": "торт", "🍰": "десерт", "🍕": "пицца",
#     "🍔": "бургер", "🍟": "картошка", "🍗": "курочка", "🍺": "пиво", "🍷": "вино",
#     "🥂": "тост", "☕": "кофе", "🍵": "чай", "🍼": "бутылочка", "🚗": "машина",
#     "🚀": "ракета", "✈️": "самолёт", "🚁": "вертолёт", "🏠": "дом", "🏢": "здание",
#     "📱": "смартфон", "💻": "ноутбук", "📷": "камера", "🎧": "наушники", "🎮": "игра",
#     "🎵": "музыка", "📚": "книги", "📖": "чтение", "📝": "запись", "💡": "идея",
#     "🔒": "замок", "🔓": "открытый замок", "🛒": "покупка", "💸": "деньги", "💰": "сумка с деньгами",
#     "🪙": "монета", "🏆": "трофей", "🥇": "золото", "🥈": "серебро", "🥉": "бронза",
#     "👑": "корона", "⚽": "футбол", "🏀": "баскетбол", "🏈": "регби", "⚾": "бейсбол",
#     "🥊": "бокс", "🏓": "пинг-понг", "⛷️": "лыжи", "🏋️": "тяжёлая атлетика", "🚴": "велосипед"
# }



# # ── реклама ───────────────────────────────────
# ADS_MARKERS = [
#     "подписывайтесь", "подпишись", "реклама", "рекламная", "рекламируем", "на правах рекламы",
#     "партнёрский", "партнерский", "спонсор", "промо", "купить", "покупай", "скидка", "промокод"
# ]

# def detect_ads(text: str) -> bool:
#     return any(marker in text.lower() for marker in ADS_MARKERS)

# def extract_emojis(text: str) -> str:
#     return " ".join(emoji.distinct_emoji_list(text))

# def replace_emojis_with_words(text: str) -> str:
#     for emo in emoji.distinct_emoji_list(text):
#         word = EMOJI_RU.get(emo, "")
#         text = text.replace(emo, f" {word} " if word else " ")
#     return text

# def normalize_text(text: str) -> str:
#     t = html.unescape(str(text)).lower()
#     t = URL_RE.sub(" ", t)
#     t = HTML_TAGS.sub(" ", t)
#     t = HASHTAG.sub(r"\1", t)
#     t = MENTION.sub(r"\1", t)
#     t = replace_emojis_with_words(t)
#     t = NON_CYR.sub(" ", t)
#     return MULTI_SP.sub(" ", t).strip()

# def lemmatize(text: str) -> str:
#     return " ".join(
#         morph.parse(w)[0].normal_form
#         for w in text.split() if w not in stopru
#     )

# def main():
#     if not RAW_CSV.exists():
#         print(f"✖ Нет файла: {RAW_CSV}", file=sys.stderr)
#         return

#     df = pd.read_csv(RAW_CSV)
#     print(f"• Загружено {len(df)} записей")

#     df["text"] = df["text"].fillna("").astype(str)
#     df["is_ad_my_classif"] = df["text"].apply(detect_ads)
#     df["emojis"] = df["text"].apply(extract_emojis)
#     df["clean_text"] = df["text"].apply(normalize_text)
#     df["lemma_text"] = df["clean_text"].apply(lemmatize)

#     df = df[df["lemma_text"].str.len() > 5]
#     df = df.drop_duplicates(subset=["lemma_text"])

#     df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
#     print(f"✓ Сохранено {len(df)} строк → {OUT_CSV}")

# if __name__ == "__main__":
#     main()
