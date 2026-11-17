import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="📉", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "bankruptcy_model.pkl"
ASSET_DIRS = [APP_DIR / "assets", APP_DIR]

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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter+Tight:wght@400;500;600&family=Syncopate:wght@400;700&display=swap');
:root{--orange:#C94B1B;--cream:#F1E9DC;--panel:#E8DFD0;--ink:#29251F;--muted:#786F65;--body:#4A4339;
--line:#D8CCBD;--bad:#B94332;--good:#376B50;--pad:2.8rem;--bar:96px;
--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace;--sans:'Inter Tight',system-ui,sans-serif}
header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.stApp,[data-testid="stMain"],[data-testid="stAppViewContainer"]{background:var(--orange);overflow:hidden!important}
.block-container,[data-testid="stMainBlockContainer"]{width:min(1280px,calc(100% - 6vw))!important;max-width:none!important;
 height:calc(100dvh - 2.4rem);margin:1.2rem auto!important;padding:0!important;gap:0!important;overflow-y:auto;overflow-x:hidden;
 background:var(--cream);border:1px solid var(--line);box-shadow:0 3px 0 rgba(0,0,0,.18);
 scrollbar-width:thin;scrollbar-color:#C4B6A2 transparent}
.block-container::-webkit-scrollbar{width:8px}
.block-container::-webkit-scrollbar-thumb{background:#C4B6A2;border-radius:8px}
.block-container::-webkit-scrollbar-track{background:transparent}
.block-container [data-testid="stVerticalBlock"]{gap:0}
html,body,.stApp,p,label,li{font-family:var(--sans);color:var(--ink)}
iframe[height="0"]{display:none}

/* fixed header and navigation */
:is([data-testid="stElementContainer"],[data-testid="element-container"]):has(.bp-bar){position:sticky;top:0;z-index:40;background:var(--cream)}
.topbar{display:flex;justify-content:space-between;align-items:center;height:50px;padding:0 var(--pad);border-bottom:1px solid var(--line);
 font-family:var(--mono);font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.topbar a.brand{font-family:var(--sans);font-size:1.02rem;letter-spacing:0;text-transform:none;font-weight:500;color:var(--ink)!important;text-decoration:none!important;border:0}
.tag{border:1px solid var(--line);padding:.35rem .7rem}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--orange);margin-right:.5rem}
.bp-bar,.topbar,.bp-nav{background:var(--cream)}
.bp-nav{display:flex;justify-content:center;height:46px;border-bottom:1px solid var(--line);overflow-x:auto;scrollbar-width:none}
.bp-nav a{display:flex;align-items:center;padding:0 1.7rem;border-left:1px solid var(--line);font-family:var(--mono);font-size:.72rem;
 letter-spacing:.14em;text-transform:uppercase;color:var(--ink)!important;text-decoration:none;border-radius:0;transition:color .2s}
.bp-nav a:first-child{border-left:0}
.bp-nav a,.bp-nav a:focus,.bp-nav a:focus-visible,.bp-nav a:active{outline:none!important}
.bp-nav a:not(.active){box-shadow:none!important}
.bp-nav a.active,.bp-nav a.active:focus{box-shadow:inset 0 -2px 0 var(--orange)!important}
.bp-nav a:hover{color:var(--orange)!important}
.bp-nav a.active{color:var(--orange)!important;box-shadow:inset 0 -2px 0 var(--orange)}

/* sections */
.anchor{height:0;scroll-margin-top:calc(var(--bar) + 4px)}
.sec{padding:0 var(--pad)}
.sec-head{display:grid;grid-template-columns:1.25fr .75fr;gap:2.5rem;align-items:center;padding:3rem 0 2rem}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
h2.title{font-family:var(--sans);font-weight:500;font-size:clamp(2.2rem,4vw,3.2rem);line-height:1.02;letter-spacing:-.035em;margin:.8rem 0 .9rem;padding:0;color:var(--ink)}
h2.title em{font-style:normal;color:var(--orange)}
.lead{font-size:1rem;line-height:1.65;color:var(--muted);max-width:34rem;margin:0}
.ill{display:flex;justify-content:center;align-items:center}
.ill img{max-height:235px;max-width:100%;object-fit:contain;display:block}
.ill img.mul{mix-blend-mode:multiply}
.part{border-top:1px solid var(--line);padding:1.9rem 0 2rem;display:grid;grid-template-columns:110px 1fr;gap:1.5rem}
.part .num{font-family:var(--mono);font-size:.8rem;letter-spacing:.1em;color:var(--orange);padding-top:.35rem}
.part h3{font-family:var(--sans);font-size:1.35rem;font-weight:500;letter-spacing:-.02em;margin:0 0 .8rem;padding:0}
.part h4{font-family:var(--mono);font-size:.72rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin:1.5rem 0 .7rem;padding:0}
.part p,.part li{font-size:.97rem;line-height:1.7;color:var(--body);margin:0 0 .75rem;max-width:46rem}
.part ul{margin:0 0 .75rem;padding-left:1.1rem}
.part li{margin:0 0 .3rem!important}
.part b{color:var(--ink);font-weight:600}
.grid{display:grid;gap:1rem;margin:1rem 0 .4rem}
.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}
.card{border:1px solid var(--line);padding:1.1rem 1.2rem;background:var(--cream)}
.card .k{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;color:var(--orange);margin-bottom:.35rem}
.card h5{font-size:1rem;font-weight:600;margin:0 0 .4rem;padding:0;color:var(--ink)}
.card p{font-size:.88rem;line-height:1.6;margin:0;color:var(--body)}
.stat{border:1px solid var(--line);padding:1rem 1.2rem}
.stat b{display:block;font-size:2rem;font-weight:500;letter-spacing:-.03em;color:var(--ink)}
.stat span{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.tblwrap{overflow-x:auto;margin:.6rem 0 .8rem;border:1px solid var(--line)}
table.tbl{width:100%;border-collapse:collapse;font-size:.9rem;min-width:520px}
.tbl th{font-family:var(--mono);font-size:.66rem;font-weight:500;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);text-align:right;padding:.7rem .9rem;border-bottom:1px solid var(--line);background:var(--panel)}
.tbl td{padding:.65rem .9rem;border-bottom:1px solid var(--line);text-align:right;color:var(--body);font-variant-numeric:tabular-nums}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.tbl tr:last-child td{border-bottom:0}
.tbl th,.tbl td{border-left:0!important;border-right:0!important;border-top:0!important}
.tbl th:first-child,.tbl td:first-child{border-left:0}
.tbl tr.hl td{color:var(--ink);font-weight:600;background:#EFE3D1}
.tbl tr.hl td:first-child{box-shadow:inset 3px 0 0 var(--orange)}
.vis{border:1px solid var(--line);background:var(--cream);padding:1.1rem}
.vis h5{font-size:.98rem;font-weight:600;margin:0 0 .3rem;padding:0}
.vis p{font-size:.86rem!important;line-height:1.55!important;margin:0 0 .8rem!important;color:var(--muted)!important}
.vis img{width:100%;height:auto;display:block;background:#fff;border:1px solid var(--line)}
.vis img.cap{max-height:420px;object-fit:contain}
.vis.w img{max-width:740px;margin:0 auto}
.bar{display:flex;height:9px;margin:.9rem 0 .5rem;border-radius:5px;overflow:hidden}
.bar i{display:block;background:var(--good)}.bar i+i{background:#C97B3E}
.note{font-size:.88rem!important;color:var(--muted)!important}
.pill{display:inline-block;font-family:var(--mono);font-size:.72rem;letter-spacing:.14em;text-transform:uppercase}

/* prediction */
.st-key-inputs{padding:0 var(--pad);border-top:1px solid var(--line)}
.st-key-inputs h3{font-family:var(--sans);font-size:1.35rem;font-weight:500;letter-spacing:-.02em;margin:1.9rem 0 1.1rem;padding:0}
.st-key-inputs div[data-testid="stHorizontalBlock"]{gap:1.6rem!important;margin-bottom:1.1rem}
@media (min-width:901px){.st-key-inputs div[data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important}.st-key-inputs div[data-testid="stColumn"]{flex:1 1 0!important}}
.st-key-inputs div[data-testid="stColumn"]{min-width:0}
[data-testid="stWidgetLabel"]{min-height:1.5rem}
[data-testid="stWidgetLabel"] p{font-family:var(--mono);font-size:.74rem;letter-spacing:.03em;color:var(--ink);white-space:nowrap}
[data-testid="stSelectbox"]>div[data-rac],div[data-baseweb="select"]{width:86%}
[data-testid="stSelectbox"] [role="group"]{background:var(--cream)!important;border:1px solid var(--line)!important;border-radius:0!important;height:44px;min-height:44px;box-shadow:none!important}
[data-testid="stSelectbox"] [role="group"]:hover,[data-testid="stSelectbox"] [role="group"]:focus-within{border-color:var(--orange)!important}
[data-testid="stSelectbox"] input{font-family:var(--sans);font-size:.9rem;color:var(--ink)}
[role="listbox"],[role="option"]{background:var(--cream)!important;font-family:var(--sans)}
[role="option"]:hover,[role="option"][data-focused="true"]{background:var(--panel)!important}
div[data-baseweb="select"]>div{background:var(--cream)!important;border:1px solid var(--line)!important;border-radius:0!important;height:44px;min-height:44px;font-size:.9rem}
div[data-baseweb="select"]>div:hover,div[data-baseweb="select"]>div:focus-within{border-color:var(--orange)!important}
ul[role="listbox"],div[data-baseweb="popover"] *{background:var(--cream);font-family:var(--sans)}
li[role="option"]:hover{background:var(--panel)!important}
.st-key-predbox{box-sizing:border-box;width:calc(100% - 2 * var(--pad))!important;margin:1.2rem var(--pad) 0;border:1px solid var(--line);padding:1.7rem 1.9rem 1.5rem}
.st-key-predbox div[data-testid="stHorizontalBlock"]{gap:2rem!important;align-items:center}
[data-testid="stButton"],.st-key-predbtn{width:100%;max-width:230px}
.stButton>button{background:var(--orange);color:var(--cream);border:1px solid var(--orange);border-radius:0;width:100%;min-height:52px;
 font-family:var(--mono);font-size:.82rem;letter-spacing:.16em;text-transform:uppercase}
.stButton>button:hover,.stButton>button:focus{background:var(--ink);border-color:var(--ink);color:var(--cream)}
.stButton>button p{color:var(--cream);font-family:var(--mono);font-size:.82rem;letter-spacing:.16em}
.res{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}
.res .l{font-family:var(--mono);font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem}
.res .v{font-size:1.7rem;font-weight:500;letter-spacing:-.02em;color:var(--ink)}
.res .v.bad{color:var(--bad)}.res .v.good{color:var(--good)}.res .v.idle{color:var(--muted)}
.msg{border-top:1px solid var(--line);margin-top:1.4rem;padding-top:1.2rem;font-size:1.02rem;line-height:1.65;color:var(--ink)}
.msg.idle{color:var(--muted)}

/* footer */
.footer{border-top:1px solid var(--line);margin-top:3rem;padding:1rem var(--pad) 1.1rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:.4rem;
 font-family:var(--mono);font-size:.64rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}

/* cover */
.cover{min-height:calc(100dvh - 2.4rem - 2px);display:flex;flex-direction:column;justify-content:space-between;padding:0 var(--pad) 2.2rem}
.cover .mid{flex:1;display:flex;align-items:center;justify-content:center;text-align:center;padding:2rem 0}
.cover h1{font-family:'Syncopate',var(--sans);font-weight:700;color:var(--orange);text-transform:uppercase;margin:0;padding:0;
 font-size:clamp(1.5rem,6vw,5.4rem);line-height:1.12;letter-spacing:-.01em}
.cover .row{display:flex;justify-content:space-between;align-items:center;gap:1.5rem;flex-wrap:wrap;border-top:1px solid var(--ink);padding-top:1.4rem}
.cover .row p{font-family:var(--mono);font-size:.95rem;color:var(--ink);margin:0}
.cover a.go{font-family:var(--mono);font-size:.78rem;letter-spacing:.16em;text-transform:uppercase;color:var(--ink)!important;text-decoration:none;
 border:1px solid var(--ink);border-radius:40px;padding:.85rem 2rem;transition:all .2s}
.cover a.go:hover{background:var(--orange);border-color:var(--orange);color:var(--cream)!important}
.cover .top{padding:1.4rem 0 0;font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}

@media (max-width:900px){
 :root{--pad:1.3rem}
 .sec-head{grid-template-columns:1fr;gap:1.2rem;padding:2rem 0 1.4rem}.ill{justify-content:flex-start}.ill img{max-height:190px}
 .part{grid-template-columns:1fr;gap:.4rem}.g3{grid-template-columns:1fr 1fr}
 .topbar .tag{display:none}.bp-nav{justify-content:flex-start}.bp-nav a{padding:0 1.1rem;white-space:nowrap}
 [data-testid="stWidgetLabel"] p{white-space:normal}div[data-baseweb="select"],[data-testid="stSelectbox"]>div[data-rac]{width:100%}
 .st-key-inputs div[data-testid="stColumn"]{flex:1 1 calc(50% - 1rem)!important;min-width:0}
 .res{grid-template-columns:1fr}.st-key-predbox{padding:1.3rem 1.2rem}
}
@media (max-width:640px){.g2,.g3{grid-template-columns:1fr}
 .st-key-inputs div[data-testid="stColumn"]{flex:1 1 100%!important}
 .block-container,[data-testid="stMainBlockContainer"]{width:100%!important;margin:0!important;height:100dvh;border:0}}
</style>
"""

NAV_JS = """
<script>
(function(){
  var P = window.parent, D = P.document;
  if (P.__bpCleanup) { try { P.__bpCleanup(); } catch (e) {} }
  var box = function(){ return D.querySelector('[data-testid="stMainBlockContainer"]') || D.querySelector('.block-container'); };
  var secs = function(){ return Array.prototype.slice.call(D.querySelectorAll('[data-sec]')); };
  function onClick(e){
    var a = e.target.closest ? e.target.closest('.bp-nav a[data-go]') : null;
    if (!a) return;
    e.preventDefault();
    var t = D.querySelector('[data-sec="' + a.getAttribute('data-go') + '"]');
    if (t) t.scrollIntoView({behavior: 'smooth', block: 'start'});
  }
  function spy(){
    var c = box(), s = secs();
    if (!c || !s.length) return;
    var line = c.getBoundingClientRect().top + 130, cur = s[0].getAttribute('data-sec');
    s.forEach(function(el){ if (el.getBoundingClientRect().top <= line) cur = el.getAttribute('data-sec'); });
    if (c.scrollTop + c.clientHeight >= c.scrollHeight - 4) cur = s[s.length - 1].getAttribute('data-sec');
    D.querySelectorAll('.bp-nav a[data-go]').forEach(function(a){
      a.classList.toggle('active', a.getAttribute('data-go') === cur);
    });
  }
  D.addEventListener('click', onClick);
  var timer = setInterval(spy, 150);
  P.__bpCleanup = function(){ D.removeEventListener('click', onClick); clearInterval(timer); };
  spy();
})();
</script>
"""


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def data_uri(name):
    for folder in ASSET_DIRS:
        for sub in ("", "charts"):
            path = folder / sub / name
            if path.is_file():
                mime = "image/png" if name.endswith(".png") else "image/jpeg"
                return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return None


def ill(name, alt, multiply=True):
    uri = data_uri(name)
    if uri is None:
        return ""
    return f'<img class="{"mul" if multiply else ""}" src="{uri}" alt="{alt}">'


def head(sec, num, title, desc, image):
    return (f'<div class="anchor" data-sec="{sec}"></div><div class="sec"><div class="sec-head"><div>'
            f'<div class="eyebrow">Section {num}</div><h2 class="title">{title}</h2><p class="lead">{desc}</p></div>'
            f'<div class="ill">{image}</div></div>')


def part(num, title, body):
    return f'<div class="part"><div class="num">{num}</div><div><h3>{title}</h3>{body}</div></div>'


def table(headers, rows, highlight=None):
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join(
        f'<tr class="{"hl" if r[0] == highlight else ""}">' + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
        for r in rows
    )
    return f'<div class="tblwrap"><table class="tbl"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def stat(value, label):
    return f'<div class="stat"><b>{value}</b><span>{label}</span></div>'


def card(k, title, text):
    return f'<div class="card"><div class="k">{k}</div><h5>{title}</h5><p>{text}</p></div>'


def vis(title, text, name, alt, cap=False, wide=False):
    uri = data_uri(name)
    img = f'<img class="{"cap" if cap else ""}" src="{uri}" alt="{alt}">' if uri else ""
    return f'<div class="vis{" w" if wide else ""}"><h5>{title}</h5><p>{text}</p>{img}</div>'


def pick(label, options, key):
    choice = st.selectbox(label, list(options), index=1, key=key)
    return options[choice]


def cover():
    st.markdown(
        '<div class="cover"><div class="top">Business risk assessment</div>'
        '<div class="mid"><h1>Bankruptcy<br>Prevention</h1></div>'
        '<div class="row"><p>A closer look at the signals that can put a business at risk.</p>'
        '<a class="go" href="?view=project" target="_self">View project</a></div></div>',
        unsafe_allow_html=True,
    )


def overview():
    html = head("overview", "01", "Overview",
                "What bankruptcy means, what this project sets out to do, and the data behind it.",
                ill("bank_building.png", "A cracked bank building surrounded by banknotes", False))
    html += part("01", "Understanding the problem",
        "<h4>What bankruptcy means</h4>"
        "<p>Bankruptcy is a situation where a company is unable to repay the debts it owes to banks, lenders or other "
        "businesses because of financial difficulties. When a business continuously struggles to meet its financial "
        "commitments, it may eventually face bankruptcy.</p>"
        "<h4>Why are we working on this?</h4>"
        "<p>The idea behind bankruptcy prevention is to identify possible warning signs of financial failure early, so "
        "businesses can understand where they are facing difficulties and take necessary action before things get worse.</p>")
    html += part("02", "About the project",
        "<h4>What are we trying to do?</h4>"
        "<p>Our project focuses on identifying whether a business is likely to face bankruptcy or not using different "
        "business-related factors.</p>"
        "<p>We use six features that describe the risks and strengths of a company to predict its financial condition. "
        "Based on these inputs, our machine learning model classifies a business into one of two categories: "
        "<b>Bankruptcy</b> or <b>Non-bankruptcy</b>.</p><p>That is the main objective of our project.</p>")
    risks = [("01", "Industrial Risk", "The possibility of a business facing problems because of the conditions or changes in the industry it operates in. This could include declining demand, increasing competition, changes in technology or new government regulations."),
             ("02", "Management Risk", "The risk caused by poor management decisions, ineffective planning or problems in the way a business is managed."),
             ("03", "Operating Risk", "The risk associated with a company's day-to-day operations. This could include operational problems that affect the normal functioning of the business.")]
    strengths = [("04", "Financial Flexibility", "A company's ability to handle unexpected financial difficulties, manage its expenses and arrange funds when needed."),
                 ("05", "Credibility", "How trustworthy and reliable a business is in the eyes of banks, investors and customers, especially when it comes to meeting its financial commitments."),
                 ("06", "Competitiveness", "How well a business can compete with other businesses operating in the same industry.")]
    html += part("03", "Getting to know our dataset",
        "<p>Before understanding how the model works, let us understand the data we are giving it.</p>"
        "<p>Our dataset contains information about 250 companies. Each row represents one company, and each company is "
        "described using six input features and one target variable.</p>"
        f'<div class="grid g3">{stat("250", "Companies")}{stat("6", "Input features")}{stat("1", "Target variable")}</div>'
        "<h4>Understanding the six features</h4><p>What information are we actually giving our model?</p>"
        "<h4>Risk-related features</h4>"
        f'<div class="grid g3">{"".join(card(*r) for r in risks)}</div>'
        "<h4>Business strength features</h4>"
        f'<div class="grid g3">{"".join(card(*s) for s in strengths)}</div>')
    html += part("04", "Understanding the target variable",
        "<p>We have discussed the six features that describe each company. But what exactly are we trying to predict?</p>"
        "<p>That is where our target variable, <b>class</b>, comes in. It contains two possible outcomes:</p>"
        f'<div class="grid g2">{card("0", "Bankruptcy", "The company belongs to the bankruptcy category.")}'
        f'{card("1", "Non-bankruptcy", "The company belongs to the non-bankruptcy category.")}</div>'
        "<p>Since our target variable has only two possible categories, this is a <b>Binary Classification</b> problem. "
        "In simple terms, we are teaching the model to identify which of these two categories a company belongs to "
        "based on its six input features.</p>")
    html += part("05", "Understanding the feature values",
        "<p>One last thing before moving into data exploration. All six features in our dataset are represented using "
        "three numerical levels:</p>"
        + table(["Level", "Meaning", "Value"], [["Low", "Level 1", "0"], ["Medium", "Level 2", "0.5"], ["High", "Level 3", "1"]])
        + "<p>These values represent the level of each feature. For risk-related features, a higher value means higher "
        "risk. For business strength features, a higher value means stronger financial flexibility, credibility or "
        "competitiveness.</p><p>This is how the information about each company is represented in our dataset and passed "
        "to the machine learning model.</p>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def exploration():
    html = head("exploration", "02", "Exploration", "Understanding the data before building the model.",
                ill("exploration.jpg", "A retro desktop computer"))
    html += part("01", "Getting familiar with the data",
        "<p>The analysis began by loading the bankruptcy dataset and examining its structure, dimensions, feature types "
        "and statistical summaries.</p><p>The dataset contains 250 records, 6 input features and 1 target variable. "
        "Initial inspection also confirmed that there were no missing values.</p>")
    html += part("02", "Missing values and duplicate records",
        "<p>There were no missing values in the dataset. However, duplicate records accounted for 58.8% of the data "
        "(147 records).</p>"
        f'<div class="grid g2">{stat("0", "Missing values")}{stat("147", "Duplicate records · 58.8%")}</div>'
        "<p>Rather than removing them immediately, I decided to retain these records, as repeated observations could "
        "carry useful information. I also wanted to ensure that these duplicates would not create data leakage during "
        "model evaluation.</p>")
    html += part("03", "Outlier detection",
        "<p>Box plots were used to examine the numerical features for potential outliers. No significant outliers were "
        "identified during this analysis.</p>")
    skew = [["Industrial risk", "-0.067"], ["Management risk", "-0.445"], ["Financial flexibility", "0.479"],
            ["Credibility", "0.113"], ["Competitiveness", "0.094"], ["Operating risk", "-0.276"]]
    html += part("04", "Understanding feature distributions",
        "<p>I examined the unique values, frequency counts and number of distinct values across the six features. "
        "All six features contained three possible levels: 0 (Low), 0.5 (Medium) and 1 (High).</p>"
        "<p>Histograms were also used to examine individual feature distributions, followed by skewness calculations.</p>"
        + table(["Feature", "Skewness"], skew)
        + "<p>The features showed relatively mild skewness, with values ranging between -0.5 and +0.5.</p>")
    html += part("05", "Visual exploration",
        "<p>To understand the dataset beyond numerical summaries, three types of visualizations were used.</p>"
        '<div class="grid g2">'
        + vis("Target distribution", "Examines the number of companies classified as bankrupt and non-bankrupt.",
              "target_distribution.png", "Bar chart of bankruptcy and non-bankruptcy counts", True)
        + vis("Correlation heatmap", "Provides an overview of relationships between the numerical features and the target variable.",
              "correlation_heatmap.png", "Correlation heatmap of the six features", True)
        + "</div>"
        '<div class="grid">'
        + vis("Feature-wise comparison", "Compares all six business factors across bankruptcy and non-bankruptcy classes, helping identify differences in their distributions.",
              "feature_comparisons.png", "Six count plots comparing features across classes", wide=True)
        + "</div>")
    html += part("06", "Preparing the target variable",
        "<p>The target variable was originally categorical, containing two classes: bankruptcy and non-bankruptcy.</p>"
        "<p>Label Encoding was applied to convert these categories into numerical labels for model training.</p>")
    html += part("07", "Handling duplicate profiles during data splitting",
        "<p>A standard train-test split was initially explored. However, because the dataset contained a high proportion "
        "of duplicate records, I investigated whether identical feature combinations could appear in both training and "
        "testing data. The analysis identified:</p>"
        f'<div class="grid g3">{stat("250", "Total records")}{stat("103", "Unique feature profiles")}{stat("0", "Profiles with conflicting labels")}</div>'
        "<p>To avoid identical feature profiles appearing in both sets, I used <b>GroupShuffleSplit</b>, grouping records "
        "based on their six input features.</p>"
        '<div class="card"><h5>Final dataset split</h5>'
        '<p><b>204</b> training records (81.6%) &nbsp;·&nbsp; <b>46</b> testing records (18.4%)</p>'
        '<div class="bar"><i style="width:81.6%"></i><i style="width:18.4%"></i></div>'
        '<p class="note">Identical feature profiles were kept together during splitting.</p></div>')
    html += ('<div class="part"><div class="num"></div><div><h3>Transition to model building</h3>'
             "<p>With the data explored, examined and prepared, the next step was to train different classification "
             "algorithms and evaluate their performance.</p></div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def evaluation():
    html = head("evaluation", "03", "Evaluation",
                "Testing different models, validating their performance and selecting the final model.",
                ill("evaluation.jpg", "A retro rotary card file"))
    html += part("01", "Feature scaling",
        "<p>Before model training, feature scaling was applied to algorithms that benefit from standardized input.</p>"
        "<ul><li><b>StandardScaler:</b> applied to Logistic Regression and SVM.</li>"
        "<li><b>Tree-based models:</b> used without feature scaling.</li></ul>")
    algos = [["01", "Logistic Regression"], ["02", "Support Vector Machine (SVM)"], ["03", "Decision Tree"],
             ["04", "Random Forest"], ["05", "Gradient Boosting"], ["06", "XGBoost"]]
    html += part("02", "Model experimentation",
        "<p>Six classification algorithms were explored:</p>" + table(["Algorithm", "Model"], algos))
    tt = [["Logistic Regression", "0.9951", "1.0000", "0.9955", "1.0000", "1.0"],
          ["SVM", "1.0000", "1.0000", "1.0000", "1.0000", "1.0"],
          ["Decision Tree", "1.0000", "1.0000", "1.0000", "1.0000", "1.0"],
          ["Random Forest", "1.0000", "1.0000", "1.0000", "1.0000", "1.0"],
          ["Gradient Boosting", "1.0000", "1.0000", "1.0000", "1.0000", "1.0"],
          ["XGBoost", "1.0000", "1.0000", "1.0000", "1.0000", "1.0"]]
    html += part("03", "Model evaluation process",
        "<p>Each model went through the same evaluation process to understand its performance.</p>"
        "<h4>Training and testing performance</h4>"
        "<p>Training and testing accuracy and F1-score were compared to see how each model performed on both datasets. "
        "The test set holds 46 records.</p>"
        + table(["Model", "Train accuracy", "Test accuracy", "Train F1", "Test F1", "Test ROC-AUC"], tt)
        + "<h4>Classification evaluation</h4><ul>"
        "<li><b>Classification Report:</b> precision, recall, F1-score and support.</li>"
        "<li><b>Confusion Matrix:</b> correct and incorrect classifications.</li>"
        "<li><b>ROC-AUC Score:</b> measures the model's ability to distinguish between the two classes.</li>"
        "<li><b>ROC Curve:</b> visualizes the trade-off between True Positive Rate and False Positive Rate.</li>"
        "<li><b>Precision-Recall Curve:</b> shows the relationship between precision and recall across different thresholds.</li></ul>"
        "<h4>Cross-validation</h4>"
        "<p>StratifiedGroupKFold was applied to evaluate model consistency across multiple folds while keeping identical "
        "feature profiles together. The original model results were compared with cross-validation results using "
        "accuracy, F1-score and ROC-AUC.</p>")
    cv = [["Logistic Regression", "0.991667", "0.989474", "0.991667", "0.990339", "0.998942"],
          ["Decision Tree", "0.986667", "0.963636", "1.000000", "0.980000", "0.990476"],
          ["Random Forest", "0.981667", "0.963636", "0.991667", "0.975745", "1.000000"],
          ["SVM", "0.983333", "0.956522", "1.000000", "0.975610", "1.000000"],
          ["XGBoost", "0.975606", "0.963636", "0.982971", "0.971300", "0.990476"],
          ["Gradient Boosting", "0.948729", "0.963636", "0.946358", "0.951292", "0.982143"]]
    html += part("04", "Comparing model performance",
        "<p>After evaluating all six algorithms, their performance was compared to identify the most suitable model for "
        "deployment. The table shows cross-validation results.</p>"
        + table(["Model", "CV accuracy", "CV precision", "CV recall", "CV F1-score", "CV ROC-AUC"], cv, "Logistic Regression")
        + '<p class="note">All six models scored 1.00 on the 46-record test set, so the test set alone could not separate '
        "them. The cross-validation results above are what set them apart.</p>")
    lrcmp = [["Accuracy", "1.0000", "0.9917"], ["F1-score", "1.0000", "0.9903"], ["ROC-AUC", "1.0000", "0.9989"]]
    html += part("05", "Final model selection",
        '<h4>Selected model</h4><p><b>Logistic Regression</b></p>'
        "<p>After comparing the performance of all six algorithms, Logistic Regression was selected as the final model "
        "for deployment. It had the highest cross-validated accuracy (0.9917), precision (0.9895) and F1-score (0.9903) "
        "of the six. Random Forest and SVM reached a cross-validated ROC-AUC of 1.00, and Decision Tree and SVM reached "
        "a recall of 1.00, so the choice rests on the whole comparison and not on one number.</p>"
        "<p>Its results were also consistent across evaluations: training accuracy 0.9951, test accuracy 1.0000 and "
        "cross-validated accuracy 0.9917.</p>"
        + table(["Metric", "Original test", "Cross-validation"], lrcmp)
        + '<p class="note">Cross-validation helps assess consistency, but it does not guarantee performance on future '
        "businesses. ROC-AUC measures how well the model separates the two classes. It is not accuracy.</p>")
    rep = [["0 (Bankruptcy)", "1.00", "1.00", "1.00", "14"], ["1 (Non-bankruptcy)", "1.00", "1.00", "1.00", "32"],
           ["Accuracy", "", "", "1.00", "46"], ["Macro average", "1.00", "1.00", "1.00", "46"],
           ["Weighted average", "1.00", "1.00", "1.00", "46"]]
    html += part("06", "Logistic Regression: detailed evaluation",
        "<h4>Classification report</h4>"
        + table(["Class", "Precision", "Recall", "F1-score", "Support"], rep)
        + '<div class="grid g3">'
        + vis("Confusion matrix", "Correct and incorrect classifications on the test set.", "logistic_confusion_matrix.png", "Confusion matrix")
        + vis("ROC curve", "ROC-AUC on the test set: 1.0.", "logistic_roc_curve.png", "ROC curve")
        + vis("Precision-Recall curve", "Average precision on the test set: 1.00.", "logistic_precision_recall_curve.png", "Precision-recall curve")
        + "</div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def prediction():
    st.markdown(
        head("prediction", "04", "Prediction",
             "Every business faces its share of risks. Some are easy to spot, while others build up over time. "
             "Understanding these factors can help us recognise early warning signs and get a clearer picture of a "
             "company's financial health.", ill("prediction.jpg", "A retro adding machine printing a paper tape"))
        + "</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="inputs"):
        st.markdown("<h3>Business Factors</h3>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            industrial_risk = pick("Industrial Risk", RISK_LEVELS, "ir")
        with c2:
            management_risk = pick("Management Risk", RISK_LEVELS, "mr")
        with c3:
            operating_risk = pick("Operating Risk", RISK_LEVELS, "or")
        c4, c5, c6 = st.columns(3)
        with c4:
            financial_flexibility = pick("Financial Flexibility", STRENGTH_LEVELS, "ff")
        with c5:
            credibility = pick("Credibility", STRENGTH_LEVELS, "cr")
        with c6:
            competitiveness = pick("Competitiveness", STRENGTH_LEVELS, "co")

    with st.container(key="predbox"):
        left, right = st.columns([1, 1.6])
        with left:
            predict = st.button("Predict", type="primary", key="predbtn")
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
                proba = model.predict_proba(X)[0]
                prob = float(proba[list(model.classes_).index(pred)])
            except FileNotFoundError:
                st.error("bankruptcy_model.pkl was not found. Place it in the same folder as this file.")
                st.stop()
            except Exception as exc:
                st.error(f"The model could not be loaded or run: {exc}")
                st.stop()
            if pred == 0:
                label, cls = "Bankruptcy", "bad"
                msg = ("The model predicts that this company is likely to face bankruptcy based on the provided business "
                       "factors. Start with a review of cash flow and short-term liabilities, then look at which operating "
                       "and management risks can be reduced. Speaking to a financial adviser early gives you more options.")
            else:
                label, cls = "Non-bankruptcy", "good"
                msg = ("The model predicts that this company is likely to remain non-bankrupt based on the provided business "
                       "factors. Keep watching cash flow, liabilities and operating conditions. The result reflects only the "
                       "six ratings above and is not a guarantee.")
            prob_txt = "&gt;99.9%" if prob > 0.9995 else f"{prob * 100:.1f}%"
        else:
            label, cls, prob_txt = "Not predicted yet", "idle", "-"
            msg = "Choose the six ratings above and press Predict."
        with right:
            st.markdown(
                f'<div class="res"><div><div class="l">Prediction</div><div class="v {cls}">{label}</div></div>'
                f'<div><div class="l">Probability</div><div class="v {"idle" if cls == "idle" else ""}">{prob_txt}</div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(f'<div class="msg {"idle" if cls == "idle" else ""}">{msg}</div>', unsafe_allow_html=True)


def shell():
    st.markdown(
        '<div class="bp-bar"><div class="topbar"><a class="brand" href="?" target="_self">Bankruptcy Prevention</a>'
        '<div><span class="tag"><span class="dot"></span>Logistic regression classifier</span></div></div>'
        '<nav class="bp-nav"><a href="#" data-go="overview" class="active">Overview</a>'
        '<a href="#" data-go="exploration">Exploration</a><a href="#" data-go="evaluation">Evaluation</a>'
        '<a href="#" data-go="prediction">Prediction</a></nav></div>',
        unsafe_allow_html=True,
    )


st.markdown(CSS, unsafe_allow_html=True)

if st.query_params.get("view") != "project":
    cover()
else:
    shell()
    overview()
    exploration()
    evaluation()
    prediction()
    st.markdown(
        '<div class="footer"><span>Bankruptcy Prevention / ML project</span>'
        '<span>Built by Harsha Ganapuram</span></div>',
        unsafe_allow_html=True,
    )
    components.html(NAV_JS, height=0)
