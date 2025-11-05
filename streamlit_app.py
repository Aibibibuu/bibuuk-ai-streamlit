# -------------------------------------------------------------
# Shelf Scanner — AI Powered Book & Goal Advisor (Bibuuk AI)
# Pro, purple UI • OpenAI GPT-4o-mini • EN/RU toggle
# -------------------------------------------------------------
import os, io, time, base64
from typing import Dict, Any
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# ------------- LOAD ENV + API KEY -------------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("❌ Missing OpenAI API key. Please add it to your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

# ------------- CONFIG / THEME -------------
PRIMARY = "#47326e"
BG = "#FAF7FF"
FONT = "Inter, -apple-system, system-ui, Segoe UI, Roboto, sans-serif"

st.set_page_config(page_title="Shelf Scanner — Book & Goal Advisor", page_icon="📚", layout="wide")

st.markdown(f"""
<style>
html,body,[class*="css"] {{ background:{BG}; font-family:{FONT}; }}
h1,h2,h3,h4,h5 {{ color:{PRIMARY}; font-weight:800; }}
.stButton>button {{
  background:{PRIMARY}!important; color:white!important; border:none!important;
  border-radius:10px!important; font-weight:600!important; padding:.65rem 1.15rem!important;
}}
.stButton>button:hover {{ background:#3a275b!important; transform: translateY(-1px); }}
.card {{
  background:#fff; border:1px solid #eee; border-radius:14px; padding:20px;
  box-shadow:0 4px 18px rgba(0,0,0,.06);
}}
.badge {{ display:inline-block; font-size:12px; color:#0b7a6f; background:#e8fffb;
  border:1px solid #bff3ea; padding:.25rem .55rem; border-radius:999px;}}
.muted {{ color:#666; }}
.divider {{ height:1px; background:#eee; margin:16px 0; }}
.right {{ text-align:right; }}
.center {{ text-align:center; }}
</style>
""", unsafe_allow_html=True)

# ------------- LANGUAGE PACK -------------
LANGS = {
    "EN": {
        "title": "Shelf Scanner: AI Powered Book & Goal Advisor",
        "subtitle": "Analyze a book or bookshelf image — and see if it helps you reach your goals.",
        "upload_hdr": "Upload a photo of your bookshelf or enter a book title manually:",
        "choose_img": "Choose an image…",
        "enter_title": "Or enter the book title and author (e.g., 'Deep Learning with Python — François Chollet')",
        "about_you": "Tell me about yourself",
        "goal": "What is your main goal right now?",
        "goal_ph": "e.g., I want to become an AI engineer and find an internship",
        "age": "How old are you?",
        "school": "What university are you studying at?",
        "year": "What year are you in?",
        "analyze_btn": "Analyze My Book Fit",
        "analysis_done": "Analysis complete!",
        "detected": "Detected Book(s):",
        "fit_hdr": "Goal Alignment & Book Recommendations:",
        "rating": "Rating:",
        "footer": "Made with 💜 by Aibiike Shainazarova · © 2025 Bibuuk AI",
    },
    "RU": {
        "title": "Shelf Scanner: AI-советник по книгам и целям",
        "subtitle": "Загрузите фото книги/полки — и узнайте, помогает ли она вашим целям.",
        "upload_hdr": "Загрузите фото книги / полки или введите название вручную:",
        "choose_img": "Выберите изображение…",
        "enter_title": "Или введите название и автора (например, «Глубокое обучение — И. Гудфеллоу»)",
        "about_you": "Расскажите о себе",
        "goal": "Какая ваша главная цель сейчас?",
        "goal_ph": "например: хочу стать AI-инженером и найти стажировку",
        "age": "Сколько вам лет?",
        "school": "В каком университете вы учитесь?",
        "year": "На каком вы курсе?",
        "analyze_btn": "Проанализировать книгу",
        "analysis_done": "Анализ завершён!",
        "detected": "Обнаруженная книга(и):",
        "fit_hdr": "Соответствие цели и рекомендации:",
        "rating": "Оценка:",
        "footer": "Сделано с 💜 Aibiike Shainazarova · © 2025 Bibuuk AI",
    },
}

# ------------- HELPERS -------------
def to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def try_detect_title_with_openai(img: Image.Image) -> str:
    """Vision title detection using GPT-4o-mini. Fallback to OCR if needed."""
    try:
        b64 = to_base64(img)
        content = [
            {"type": "text", "text": "Read the book cover and return ONLY the most likely title and author as plain text."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            if lines:
                lines.sort(key=len, reverse=True)
                return lines[0][:120]
        except Exception:
            pass
        return ""

def ask_bibuuk_analysis(title: str, form: Dict[str, Any], lang: str) -> str:
    locale = "English" if lang == "EN" else "Russian"
    prompt = f"""
You are Bibuuk AI — a professional, warm book mentor.
User language: {locale}.
Book: "{title or "(unknown)"}"

User profile:
- Goal: {form.get("goal") or ""}
- Age: {form.get("age") or ""}
- University: {form.get("school") or ""}
- Year: {form.get("year") or ""}

Write a structured answer in {locale} with headings exactly like:

# Goal Alignment & Book Recommendations:
Rating: <Good fit / Partial fit / Not the best fit>.

<2–3 short paragraphs explaining WHY the book does or doesn’t fit the goal.>

Recommended Books:
1. **<Title>** — <reason>.
2. **<Title>** — <reason>.
3. **<Title>** — <reason>.
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return resp.choices[0].message.content

# ============ UI ============
with st.sidebar:
    st.markdown("### Session Info")
    st.info("Session automatically saved (simulation).")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(f"<span class='muted'>{LANGS['EN']['footer']}</span>", unsafe_allow_html=True)

col_l, col_c, col_r = st.columns([1,2,1])
with col_l:
    lang = st.radio("Language", ["EN", "RU"], horizontal=True)

T = LANGS[lang]

with col_c:
    try:
        logo = Image.open("assets/bibuuk_logo.jpeg")
        st.image(logo, width=140)
    except Exception:
        pass
    st.markdown(f"<h1 class='center'>{T['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='center muted'>{T['subtitle']}</p>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(f"### {T['upload_hdr']}")
uploader_col, title_col = st.columns([1,1])
detected_title = ""
uploaded_image = None

with uploader_col:
    img_file = st.file_uploader(T["choose_img"], type=["jpg","jpeg","png"])
    if img_file:
        uploaded_image = Image.open(io.BytesIO(img_file.read())).convert("RGB")
        st.image(uploaded_image, caption="Preview", use_column_width=True)
        with st.spinner("Reading cover…"):
            detected_title = try_detect_title_with_openai(uploaded_image)

with title_col:
    title_input = st.text_input(T["enter_title"], value=detected_title)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown(f"### {T['about_you']}")
c1, c2 = st.columns([2,1])
with c1:
    goal = st.text_input(T["goal"], placeholder=T["goal_ph"])
    school = st.text_input(T["school"], placeholder="e.g., National Louis University")
with c2:
    age = st.number_input(T["age"], min_value=10, max_value=80, value=19, step=1)
    year = st.selectbox(T["year"], ["Freshman", "Sophomore", "Junior", "Senior", "Graduate"])

st.write("")
analyze = st.button(T["analyze_btn"])

if analyze:
    if not (title_input or uploaded_image):
        st.warning("Please upload a cover or enter a book title.")
    else:
        with st.spinner("Analyzing…"):
            result = ask_bibuuk_analysis(title_input, {"goal": goal, "age": age, "school": school, "year": year}, lang)
            time.sleep(0.8)

        st.success(T["analysis_done"])
        st.markdown(f"### {T['detected']}")
        if title_input:
            st.write(f"The visible book is titled **\"{title_input}\"**.")
        else:
            st.write("A book was detected but title was not recognized.")
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown(f"## {T['fit_hdr']}")
        st.markdown(result)

st.markdown(f"<p class='center muted' style='margin-top:32px;'>{T['footer']}</p>", unsafe_allow_html=True)
