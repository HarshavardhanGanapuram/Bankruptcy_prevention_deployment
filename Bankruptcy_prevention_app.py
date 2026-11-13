import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="📉", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bankruptcy_model.pkl"
IMAGE_NAME = "bank_building.png"
# The image may sit next to this file or inside an "assets" folder
IMAGE_CANDIDATES = [APP_DIR / IMAGE_NAME, APP_DIR / "assets" / IMAGE_NAME]

# Exact training order
FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]
# Target encoding: 0 = Bankruptcy, 1 = Non-bankruptcy

RISK_LEVELS = {"Low risk": 0.0, "Medium risk": 0.5, "High risk": 1.0}
STRENGTH_LEVELS = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter+Tight:wght@400;500;600&display=swap');

:root{--orange:#C94B1B;--cream:#F1E9DC;--panel:#E8DFD0;--ink:#29251F;--muted:#786F65;
--line:#D8CCBD;--bad:#B94332;--good:#376B50;--pad:2.6rem;
--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter Tight',system-ui,sans-serif}

header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"]{display:none!important}
.stApp{background:var(--orange)}
.block-container,[data-testid="stMainBlockContainer"]{width:min(1260px,calc(100% - 8vw))!important;max-width:none!important;
 padding:0 0 .05px!important;margin:3.2rem auto;background:var(--cream);border:1px solid var(--line);
 box-shadow:0 3px 0 rgba(0,0,0,.18);gap:0!important}
.block-container [data-testid="stVerticalBlock"]{gap:0}
html,body,.stApp,p,label,li{font-family:var(--sans);color:var(--ink)}

/* top bar */
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);
 padding:0 var(--pad);height:50px;font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.brand{font-family:var(--sans);font-size:1rem;letter-spacing:0;text-transform:none;color:var(--ink);font-weight:500}
.brand small{font-family:var(--mono);font-size:.65rem;color:var(--muted);margin-left:.5rem;letter-spacing:.08em}
.tag{border:1px solid var(--line);padding:.35rem .7rem}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--orange);margin-right:.5rem}

/* top section: title left, image right */
div[data-testid="stHorizontalBlock"]:has(.st-key-left){gap:0!important;align-items:stretch}
.st-key-left{padding:2.4rem var(--pad) 2rem;display:flex;flex-direction:column;justify-content:center;height:100%}
div[data-testid="stColumn"]:has(.st-key-right){background:var(--cream);border-left:1px solid var(--line)}
.st-key-right{padding:1.2rem 2rem 1.4rem}

.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
h1.hero{font-family:var(--sans);font-weight:500;font-size:clamp(2.4rem,4.4vw,3.6rem);line-height:1;
 letter-spacing:-.035em;margin:.9rem 0 1rem;padding:0;color:var(--ink)}
h1.hero em{font-style:normal;color:var(--orange)}
.intro{font-size:.98rem;line-height:1.65;color:var(--muted);max-width:30rem;margin:0}
.intro b{color:var(--ink);font-weight:600}
.fig{display:flex;align-items:center;justify-content:center;height:290px;margin-top:.7rem}
.fig img{max-width:100%;max-height:100%;object-fit:contain;display:block;background:transparent;border:0;box-shadow:none}

/* middle section: inputs */
.st-key-mid{padding:1.4rem var(--pad) 0;border-top:1px solid var(--line)}
.section h2{font-family:var(--sans);font-size:1.15rem;font-weight:500;margin:0 0 .9rem;padding:0}
.st-key-mid div[data-testid="stHorizontalBlock"]{gap:1.4rem!important;margin-bottom:1rem}
.st-key-mid div[data-testid="stColumn"]{min-width:0}
[data-testid="stWidgetLabel"]{min-height:1.5rem;align-items:center}
[data-testid="stWidgetLabel"] p{font-family:var(--mono);font-size:.74rem;letter-spacing:.03em;color:var(--ink);white-space:nowrap}
div[data-baseweb="select"]>div{background:var(--cream)!important;border:1px solid var(--line)!important;border-radius:0!important;
 height:44px;min-height:44px;font-size:.9rem}
div[data-baseweb="select"]>div:hover,div[data-baseweb="select"]>div:focus-within{border-color:var(--orange)!important}
ul[role="listbox"],div[data-baseweb="popover"] *{background:var(--cream);font-family:var(--sans)}
li[role="option"]:hover{background:var(--panel)!important}

/* button */
.stButton>button{background:var(--orange);color:var(--cream);border:1px solid var(--orange);border-radius:0;
 font-family:var(--mono);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;padding:.7rem 2.6rem;margin-top:.2rem}
.stButton>button:hover,.stButton>button:focus{background:var(--ink);border-color:var(--ink);color:var(--cream)}
.stButton>button p{color:var(--cream);font-family:var(--mono);font-size:.78rem;letter-spacing:.12em}

/* result */
.result{border:1px solid var(--line);border-left:4px solid var(--line);background:var(--cream);padding:1rem 1.3rem;margin:1.2rem 0 0;min-height:104px}
.result .label{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.result .class{font-size:1.45rem;font-weight:500;letter-spacing:-.02em;margin:.2rem 0 .45rem}
.result p{font-size:.93rem;line-height:1.6;color:var(--ink);margin:0;max-width:60rem}
.result.bad{border-left-color:var(--bad)}.result.bad .class{color:var(--bad)}
.result.good{border-left-color:var(--good)}.result.good .class{color:var(--good)}
.result.idle .class{color:var(--muted);font-size:1.05rem;font-weight:400}

.footer{border-top:1px solid var(--line);margin-top:1.8rem;padding:1rem var(--pad) 1.1rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem;
 font-family:var(--mono);font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}

@media (max-width:900px){
 :root{--pad:1.3rem}
 .st-key-left{padding:1.6rem var(--pad)}.st-key-right{padding:1rem var(--pad)}
 div[data-testid="stColumn"]:has(.st-key-right){border-left:0;border-top:1px solid var(--line)}
 .fig{height:auto}.topbar .tag{display:none}
 [data-testid="stWidgetLabel"] p{white-space:normal}
}
@media (max-width:640px){.block-container,[data-testid="stMainBlockContainer"]{width:100%!important;margin:0}}
</style>
"""

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def find_image():
    for candidate in IMAGE_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


@st.cache_data
def encode_image(path_str):
    return base64.b64encode(Path(path_str).read_bytes()).decode()


def pick(label, options, key, help_text):
    choice = st.selectbox(label, list(options), index=1, key=key, help=help_text)
    return options[choice]


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="topbar"><div class="brand">Bankruptcy Prevention<small>ML.01</small></div>'
    '<div><span class="tag"><span class="dot"></span>Logistic regression classifier</span></div></div>',
    unsafe_allow_html=True,
)

# Top section: title and description on the left, image on the right
left, right = st.columns([1.1, 1])

with left:
    with st.container(key="left"):
        st.markdown(
            '<div class="eyebrow">Business risk assessment</div>'
            '<h1 class="hero">Bankruptcy<br><em>Prevention</em></h1>'
            '<p class="intro">This tool estimates whether a business is likely to face bankruptcy. '
            'Rate six factors, from industrial risk to competitiveness, and the model '
            '<b>classifies the profile as bankruptcy or non-bankruptcy</b>. '
            'It was trained on records of 250 companies.</p>',
            unsafe_allow_html=True,
        )

with right:
    with st.container(key="right"):
        st.markdown('<div class="eyebrow">Fig. 01 / Business under pressure</div>', unsafe_allow_html=True)
        image_path = find_image()
        if image_path is None:
            st.warning(
                f"Illustration not found. Add {IMAGE_NAME} next to this file or inside an 'assets' folder "
                "and commit it to your repository (file names are case-sensitive on Streamlit Cloud)."
            )
        else:
            try:
                st.markdown(
                    '<div class="fig"><img alt="A cracked bank building surrounded by banknotes" '
                    f'src="data:image/png;base64,{encode_image(str(image_path))}"></div>',
                    unsafe_allow_html=True,
                )
            except OSError as exc:
                st.warning(f"The illustration could not be read: {exc}")

# Middle section: six business factors in an even 3 x 2 grid
with st.container(key="mid"):
    st.markdown('<div class="section"><h2>Business Factors</h2></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        industrial_risk = pick("Industrial Risk", RISK_LEVELS, "ir",
                               "Risk that comes from the industry the business operates in.")
    with c2:
        management_risk = pick("Management Risk", RISK_LEVELS, "mr",
                               "Risk linked to the quality of management and decisions.")
    with c3:
        operating_risk = pick("Operating Risk", RISK_LEVELS, "or",
                              "Risk in day-to-day operations.")

    c4, c5, c6 = st.columns(3)
    with c4:
        financial_flexibility = pick("Financial Flexibility", STRENGTH_LEVELS, "ff",
                                     "How easily the business can raise funds or adjust its finances. Higher is stronger.")
    with c5:
        credibility = pick("Credibility", STRENGTH_LEVELS, "cr",
                           "Standing with lenders, suppliers and customers. Higher is stronger.")
    with c6:
        competitiveness = pick("Competitiveness", STRENGTH_LEVELS, "co",
                               "Strength of the business against its competitors. Higher is stronger.")

    predict = st.button("Predict", type="primary")

    if predict:
        try:
            model = load_model()
            values = {
                "industrial_risk": industrial_risk,
                "management_risk": management_risk,
                "financial_flexibility": financial_flexibility,
                "credibility": credibility,
                "competitiveness": competitiveness,
                "operating_risk": operating_risk,
            }
            X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
            pred = int(model.predict(X)[0])
        except FileNotFoundError:
            st.error("bankruptcy_model.pkl was not found. Place it in the same folder as this file.")
            st.stop()
        except Exception as exc:
            st.error(f"The model could not be loaded or run: {exc}")
            st.stop()

        if pred == 0:
            st.markdown(
                '<div class="result bad"><div class="label">Prediction</div>'
                '<div class="class">Bankruptcy</div>'
                '<p>This profile is classified as bankruptcy. Start with a review of cash flow and '
                'short-term liabilities, then look at which operating and management risks can be '
                'reduced. Speaking to a financial adviser early gives you more options.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result good"><div class="label">Prediction</div>'
                '<div class="class">Non-bankruptcy</div>'
                '<p>This profile is classified as non-bankruptcy. Keep watching cash flow, '
                'liabilities and operating conditions. The result reflects only the six ratings '
                'above and is not a guarantee.</p></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="result idle"><div class="label">Prediction</div>'
            '<div class="class">Choose your ratings, then press Predict.</div></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer"><span>Bankruptcy Prevention / ML project</span>'
    '<span>Built with Streamlit and scikit-learn</span></div>',
    unsafe_allow_html=True,
)
