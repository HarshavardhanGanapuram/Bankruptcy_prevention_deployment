from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="📉", layout="wide")

MODEL_PATH = Path(__file__).parent / "bankruptcy_model.pkl"

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
--line:#D8CCBD;--bad:#B94332;--good:#376B50;
--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter Tight',system-ui,sans-serif}

header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"]{display:none!important}
.stApp{background:var(--orange)}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1180px!important;padding:0!important;
 margin:2.2rem auto;background:var(--cream);border:1px solid var(--line);box-shadow:0 3px 0 rgba(0,0,0,.18)}
html,body,.stApp,p,label,li{font-family:var(--sans);color:var(--ink)}

/* top bar */
.topbar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);
 padding:0 1.6rem;height:48px;font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.brand{font-family:var(--sans);font-size:1rem;letter-spacing:0;text-transform:none;color:var(--ink);font-weight:500}
.brand small{font-family:var(--mono);font-size:.65rem;color:var(--muted);margin-left:.5rem;letter-spacing:.08em}
.tag{border:1px solid var(--line);padding:.35rem .7rem}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--orange);margin-right:.5rem}

/* two columns */
div[data-testid="stHorizontalBlock"]:has(.st-key-left){gap:0!important}
.st-key-left{padding:2rem 2.4rem 1.6rem}
div[data-testid="stColumn"]:has(.st-key-right){background:var(--panel);border-left:1px solid var(--line)}
.st-key-right{padding:1.6rem 1.8rem}

.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
h1.hero{font-family:var(--sans);font-weight:500;font-size:clamp(2.3rem,4.6vw,3.5rem);line-height:1;
 letter-spacing:-.035em;margin:1rem 0 1.1rem;padding:0;color:var(--ink)}
h1.hero em{font-style:normal;color:var(--orange)}
.intro{font-size:.95rem;line-height:1.65;color:var(--muted);max-width:34rem;margin:0 0 1.3rem}
.intro b{color:var(--ink);font-weight:600}
.stats{display:inline-flex;border:1px solid var(--line);margin-bottom:1.6rem;flex-wrap:wrap}
.stats div{padding:.55rem .9rem;border-right:1px solid var(--line);font-family:var(--mono);font-size:.78rem}
.stats div:last-child{border-right:0}
.stats span{display:block;font-size:.6rem;letter-spacing:.1em;color:var(--muted);margin-bottom:.15rem}
.section{display:flex;align-items:center;gap:.8rem;border-top:1px solid var(--line);padding-top:1rem;margin:.2rem 0 .8rem}
.section h2{font-family:var(--sans);font-size:1.15rem;font-weight:500;margin:0;padding:0}
.group{font-family:var(--mono);font-size:.65rem;letter-spacing:.1em;text-transform:uppercase;color:var(--orange);margin:.5rem 0 .1rem}

/* inputs */
[data-testid="stWidgetLabel"] p{font-family:var(--mono);font-size:.74rem;letter-spacing:.03em;color:var(--ink)}
div[data-baseweb="select"]>div{background:var(--cream)!important;border:1px solid var(--line)!important;border-radius:0!important;
 min-height:42px;font-size:.9rem}
div[data-baseweb="select"]>div:hover,div[data-baseweb="select"]>div:focus-within{border-color:var(--orange)!important}
ul[role="listbox"],div[data-baseweb="popover"] *{background:var(--cream);font-family:var(--sans)}
li[role="option"]:hover{background:var(--panel)!important}

/* button */
.stButton>button{background:var(--orange);color:var(--cream);border:1px solid var(--orange);border-radius:0;
 font-family:var(--mono);font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;padding:.7rem 2.4rem;margin-top:.8rem}
.stButton>button:hover,.stButton>button:focus{background:var(--ink);border-color:var(--ink);color:var(--cream)}
.stButton>button p{color:var(--cream);font-family:var(--mono);font-size:.78rem;letter-spacing:.12em}

/* result */
.result{border:1px solid var(--line);border-left:4px solid var(--line);background:var(--cream);padding:1rem 1.2rem;margin-top:1.2rem}
.result .label{font-family:var(--mono);font-size:.66rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.result .class{font-size:1.5rem;font-weight:500;letter-spacing:-.02em;margin:.2rem 0 .5rem}
.result p{font-size:.92rem;line-height:1.6;color:var(--ink);margin:0}
.result.bad{border-left-color:var(--bad)}.result.bad .class{color:var(--bad)}
.result.good{border-left-color:var(--good)}.result.good .class{color:var(--good)}
.result.idle .class{color:var(--muted);font-size:1.05rem;font-weight:400}

.fig svg{width:100%;height:auto;display:block;margin-top:.8rem}
.caption{display:flex;justify-content:space-between;font-family:var(--mono);font-size:.62rem;letter-spacing:.08em;
 text-transform:uppercase;color:var(--muted);margin-top:.8rem;border-top:1px solid var(--line);padding-top:.7rem}
.footer{border-top:1px solid var(--line);padding:.8rem 1.6rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem;
 font-family:var(--mono);font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}

@media (max-width:900px){
 .st-key-left{padding:1.5rem 1.2rem}.st-key-right{padding:1.2rem}
 div[data-testid="stColumn"]:has(.st-key-right){border-left:0;border-top:1px solid var(--line)}
 .topbar .tag{display:none}
}
@media (max-width:640px){.block-container,[data-testid="stMainBlockContainer"]{margin:0}}
</style>
"""

ILLUSTRATION = """
<svg viewBox="0 0 480 400" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A businessman sitting on the floor beside a falling chart">
<defs><pattern id="g" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M40 0H0V40" fill="none" stroke="#D8CCBD" stroke-width="1"/></pattern></defs>
<rect x="20" y="20" width="440" height="360" fill="url(#g)" stroke="#D8CCBD"/>
<g fill="none" stroke="#B9AE9C" stroke-width="1.4"><circle cx="90" cy="62" r="22"/><circle cx="410" cy="80" r="28"/><circle cx="58" cy="300" r="20"/><circle cx="440" cy="250" r="14"/></g>
<g font-family="Georgia,serif" fill="#B9AE9C" text-anchor="middle" font-size="22"><text x="90" y="70">$</text><text x="410" y="89" font-size="28">$</text><text x="58" y="308" font-size="20">$</text><text x="440" y="255" font-size="14">$</text></g>
<g fill="#F7F1E6" stroke="#29251F" stroke-width="1.2" stroke-linejoin="round">
<polygon points="300,40 338,48 330,86 292,78" transform="rotate(12 315 60)"/>
<polygon points="95,150 125,156 118,186 88,180"/>
<polygon points="370,170 402,180 392,206 360,198"/>
<polygon points="150,350 190,338 214,356 170,368"/>
<polygon points="270,356 310,352 322,368 280,374"/></g>
<g transform="translate(10,10)" fill="#B9AE9C" stroke="none"><path d="M50 80 L170 170 L230 120 L368 258 L394 232 L400 290 L342 284 L368 258Z" opacity="0"/></g>
<g stroke-linejoin="miter">
<path d="M58 88 L178 178 L238 128 L376 266" fill="none" stroke="#9B9488" stroke-width="26"/>
<polygon points="408,298 402,240 350,292" fill="#9B9488"/>
<path d="M50 80 L170 170 L230 120 L368 258" fill="none" stroke="#C94B1B" stroke-width="26"/>
<polygon points="400,290 394,232 342,284" fill="#C94B1B"/></g>
<ellipse cx="270" cy="372" rx="120" ry="7" fill="#D8CCBD"/>
<g stroke="#29251F" stroke-linecap="round" stroke-linejoin="round" fill="none">
<path d="M250 308 L320 332 L322 362" stroke-width="16"/>
<path d="M250 308 L300 298 L346 346" stroke-width="16"/>
<path d="M250 268 L292 292 L312 284" stroke="#29251F" stroke-width="9"/>
<path d="M240 270 L222 320 L206 350" stroke-width="9"/></g>
<polygon points="228,258 262,252 272,306 232,314" fill="#F7F1E6" stroke="#29251F" stroke-width="1.5" stroke-linejoin="round"/>
<polygon points="247,258 255,258 258,290 251,298 244,290" fill="#C94B1B"/>
<circle cx="240" cy="238" r="17" fill="#F7F1E6" stroke="#29251F" stroke-width="1.5"/>
<path d="M222 236 Q226 216 246 220 Q258 224 258 234 Q244 228 232 238Z" fill="#29251F"/>
<polygon points="380,350 394,344 402,354 388,360" fill="#C94B1B" stroke="#29251F" stroke-width="1.2"/>
</svg>
"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def pick(label, options, key, help_text):
    choice = st.selectbox(label, list(options), index=1, key=key, help=help_text)
    return options[choice]


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="topbar"><div class="brand">Bankruptcy Prevention<small>ML.01</small></div>'
    '<div><span class="tag"><span class="dot"></span>Logistic regression classifier</span></div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.1, 1])

with left:
    with st.container(key="left"):
        st.markdown(
            '<div class="eyebrow">Business risk assessment / Classification model</div>'
            '<h1 class="hero">Bankruptcy<br><em>Prevention</em></h1>'
            '<p class="intro">This tool estimates whether a business is likely to face bankruptcy. '
            'Rate six factors, from industrial risk to competitiveness, and the model '
            '<b>classifies the profile as bankruptcy or non-bankruptcy</b>. '
            'It was trained on records of 250 companies.</p>'
            '<div class="stats"><div><span>MODEL</span>Logistic Reg.</div>'
            '<div><span>INPUTS</span>6 factors</div><div><span>DATA</span>250 firms</div></div>'
            '<div class="section"><h2>Business Factors</h2></div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="group">Risks</div>', unsafe_allow_html=True)
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

        st.markdown('<div class="group">Strengths</div>', unsafe_allow_html=True)
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

with right:
    with st.container(key="right"):
        st.markdown(
            '<div class="eyebrow">Fig. 01 / Business under pressure</div>'
            f'<div class="fig">{ILLUSTRATION}</div>'
            '<div class="caption"><span>Falling revenue</span><span>Rising liabilities</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer"><span>Bankruptcy Prevention / ML project</span>'
    '<span>Built with Streamlit and scikit-learn</span></div>',
    unsafe_allow_html=True,
)
