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
.sec-head{display:grid;grid-template-columns:1.15fr .85fr;gap:2.25rem;align-items:center;padding:3rem 0 2rem}
.eyebrow{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
h2.title{font-family:var(--sans);font-weight:500;font-size:clamp(2.2rem,4vw,3.2rem);line-height:1.02;letter-spacing:-.035em;margin:.8rem 0 .9rem;padding:0;color:var(--ink)}
h2.title em{font-style:normal;color:var(--orange)}
.lead{font-size:1rem;line-height:1.65;color:var(--muted);max-width:34rem;margin:0}
.ill{display:flex;justify-content:center;align-items:center;min-height:250px;background:transparent;overflow:visible}
/* Illustrations sit directly on the cream canvas: no frame, white tile or shadow. */
.ill img{display:block;width:auto;height:auto;max-width:94%;max-height:330px;object-fit:contain;
 background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;
 mix-blend-mode:multiply}
.ill img.mul{mix-blend-mode:multiply}
.ill-missing{font-family:var(--mono);font-size:.68rem;letter-spacing:.06em;color:var(--muted);border:1px dashed var(--line);padding:1rem 1.2rem}
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
.vis img{width:100%;height:320px;object-fit:contain;display:block;background:#fff;border:1px solid var(--line);padding:.35rem}
.vis img.cap{height:380px;max-height:500px;object-fit:contain}
.vis.w img{max-width:1000px;height:430px;margin:0 auto}
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
[data-testid="stSelectbox"] [role="group"]{background:#fff!important;border:1px solid var(--line)!important;border-radius:0!important;height:44px;min-height:44px;box-shadow:none!important}
[data-testid="stSelectbox"] [role="group"]:hover,[data-testid="stSelectbox"] [role="group"]:focus-within{background:#fff!important;border-color:var(--orange)!important}
[data-testid="stSelectbox"] input{background:#fff!important;font-family:var(--sans);font-size:.9rem;color:var(--ink)!important}
[role="listbox"],[role="option"]{background:#fff!important;color:var(--ink)!important;font-family:var(--sans)}
[role="option"]:hover,[role="option"][data-focused="true"]{background:#fff!important;color:var(--ink)!important}
div[data-baseweb="select"]>div{background:#fff!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:0!important;height:44px;min-height:44px;font-size:.9rem}
div[data-baseweb="select"]>div:hover,div[data-baseweb="select"]>div:focus-within{background:#fff!important;border-color:var(--orange)!important}
ul[role="listbox"],div[data-baseweb="popover"] *{background:#fff!important;color:var(--ink)!important;font-family:var(--sans)}
li[role="option"]:hover{background:#fff!important;color:var(--ink)!important}
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
 .sec-head{grid-template-columns:1fr;gap:1.2rem;padding:2rem 0 1.4rem}.ill{justify-content:flex-start}.ill{min-height:190px}.ill img{max-height:245px;max-width:100%}
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
        for sub in ("", "illustrations", "images", "charts"):
            path = folder / sub / name
            if path.is_file():
                import mimetypes
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()
    return None


def ill(name, alt, multiply=True):
    uri = data_uri(name)
    if uri is None:
        return ""
    return f'<img class="{"mul" if multiply else ""}" src="{uri}" alt="{alt}">'


def head(sec, num, title, desc, image, show_title=True):
    title_html = (
        f'<div class="eyebrow">Section {num}</div><h2 class="title">{title}</h2>'
        if show_title else ""
    )
    return (
        f'<div class="anchor" data-sec="{sec}"></div><div class="sec">'
        f'<div class="sec-head"><div>{title_html}<p class="lead">{desc}</p></div>'
        f'<div class="ill">{image}</div></div>'
    )


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
    html = head(
        "overview", "01", "Overview",
        "A practical look at the business signals in this project and what the model is designed to identify.",
        ill("bank_building.png", "A cracked bank building surrounded by banknotes"),
        show_title=False,
    )

    html += part("01", "Why bankruptcy signals matter",
        "<p>When a business struggles to meet its financial commitments, the pressure can build over time. "
        "Looking at key business factors can help identify warning signs before the situation becomes more serious.</p>"
        "<ul><li>Understand where a business may be exposed to risk.</li>"
        "<li>Look at the strengths that may help it handle financial pressure.</li>"
        "<li>Use those signals to estimate whether the business falls into the bankruptcy or non-bankruptcy class.</li></ul>")

    html += part("02", "What this project does",
        "<p>This project uses six business factors to classify a company into one of two outcomes:</p>"
        '<div class="grid g2">'
        + card("0", "Bankruptcy", "The model assigns the company to the bankruptcy class.")
        + card("1", "Non-bankruptcy", "The model assigns the company to the non-bankruptcy class.")
        + "</div>"
        "<p>The model works with the ratings supplied for each factor. It does not use company names, financial statements "
        "or live business records.</p>")

    risks = [
        ("01", "Industrial Risk", "Pressure caused by changes in the industry, such as falling demand, stronger competition, technology shifts or regulation."),
        ("02", "Management Risk", "Problems linked to business decisions, planning or the way the company is managed."),
        ("03", "Operating Risk", "Difficulties in day-to-day operations that can affect how the business functions.")
    ]
    strengths = [
        ("04", "Financial Flexibility", "The company's ability to manage expenses, arrange funds and handle unexpected financial pressure."),
        ("05", "Credibility", "How reliably the company is viewed by lenders, investors and customers, especially when meeting commitments."),
        ("06", "Competitiveness", "How well the company can hold its position against others in the same industry.")
    ]
    html += part("03", "The six business factors",
        "<p>The model looks at three risk factors and three business-strength factors.</p>"
        "<h4>Risk factors</h4>"
        f'<div class="grid g3">{"".join(card(*r) for r in risks)}</div>'
        "<h4>Business strengths</h4>"
        f'<div class="grid g3">{"".join(card(*v) for v in strengths)}</div>')

    html += part("04", "The dataset at a glance",
        f'<div class="grid g3">{stat("250", "Company records")}{stat("6", "Input features")}{stat("1", "Target column")}</div>'
        "<ul><li>Each row represents one company.</li>"
        "<li>The six input features describe its risk and business strengths.</li>"
        "<li>The target column, <b>class</b>, records the outcome: bankruptcy or non-bankruptcy.</li></ul>")

    html += part("05", "How the feature ratings work",
        "<p>Each feature is recorded using one of three values:</p>"
        + table(["Rating", "Value"], [["Low", "0"], ["Medium", "0.5"], ["High", "1"]])
        + "<ul><li>For the three risk factors, a higher value means a higher level of risk.</li>"
        "<li>For the three business-strength factors, a higher value means a stronger position.</li></ul>"
        "<p>The same number therefore has a different interpretation depending on the feature being rated.</p>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def exploration():
    html = head(
        "exploration", "02", "Exploration",
        "A closer look at the records, feature values and checks carried out before model training.",
        ill("exploration.jpg", "A retro desktop computer"),
        show_title=False,
    )
    html += part("01", "First look at the data",
        "<ul><li><b>250 records</b> describing companies.</li>"
        "<li><b>6 input features</b> and <b>1 target variable</b>.</li>"
        "<li><b>No missing values</b> were found during the initial check.</li></ul>"
        "<p>The first step was to inspect the dataset's dimensions, data types, unique values and statistical summaries.</p>")

    html += part("02", "Duplicate records",
        f'<div class="grid g2">{stat("147", "Duplicate records · 58.8%")}{stat("0", "Missing values")}</div>'
        "<ul><li>Duplicate rows were retained rather than removed.</li>"
        "<li>There were 103 unique feature profiles across the 250 records.</li>"
        "<li>None of those profiles had conflicting target labels.</li></ul>"
        "<p>Because repeated profiles could appear in both training and testing data, the split was designed to keep "
        "identical feature combinations together.</p>")

    html += part("03", "Outlier and distribution checks",
        "<ul><li>Box plots were used to inspect the numerical features.</li>"
        "<li>No significant outliers were identified.</li>"
        "<li>Each of the six features had three possible values: 0, 0.5 and 1.</li>"
        "<li>Histograms and frequency counts were used to understand how those values were distributed.</li></ul>")

    skew = [["Industrial risk", "-0.067"], ["Management risk", "-0.445"],
            ["Financial flexibility", "0.479"], ["Credibility", "0.113"],
            ["Competitiveness", "0.094"], ["Operating risk", "-0.276"]]
    html += part("04", "Skewness across the features",
        "<p>Skewness was calculated to check whether each feature's distribution leaned to one side.</p>"
        + table(["Feature", "Skewness"], skew)
        + "<p>The values are relatively close to zero, indicating no strong skew in these feature distributions.</p>")

    html += part("05", "What the visualizations show",
        '<div class="grid g2">'
        + vis("Target distribution", "The number of companies in each target class.",
              "target_distribution.png", "Bar chart of bankruptcy and non-bankruptcy counts", True)
        + vis("Correlation heatmap", "How the numerical features and target relate to one another.",
              "correlation_heatmap.png", "Correlation heatmap of the six features", True)
        + "</div>"
        '<div class="grid">'
        + vis("Feature-wise comparison", "A side-by-side view of the six feature levels across the two target classes.",
              "feature_comparisons.png", "Six count plots comparing features across classes", wide=True)
        + "</div>")

    html += part("06", "Preparing the target and splitting the data",
        "<ul><li>The categorical target was label-encoded into numerical classes.</li>"
        "<li><b>GroupShuffleSplit</b> was used with the six input features as groups.</li>"
        "<li>This kept identical feature profiles together instead of allowing copies of the same profile into both sets.</li></ul>"
        f'<div class="grid g3">{stat("250", "Total records")}{stat("204", "Training records · 81.6%")}{stat("46", "Testing records · 18.4%")}</div>'
        '<div class="card"><h5>Final split</h5><div class="bar"><i style="width:81.6%"></i><i style="width:18.4%"></i></div>'
        '<p class="note">Training and testing sets contain separate feature profiles.</p></div>')

    html += part("07", "Ready for model building",
        "<p>With the data checked, explored and split without separating duplicate profiles, the next step was to train "
        "and compare classification models.</p>")
    st.markdown(html + "</div>", unsafe_allow_html=True)


def evaluation():
    html = head(
        "evaluation", "03", "Evaluation",
        "Six classification models were compared using a held-out test set and grouped cross-validation.",
        ill("evaluation.jpg", "A retro rotary card file"),
        show_title=False,
    )
    html += part("01", "Preparing the features",
        "<ul><li><b>Logistic Regression</b> and <b>SVM</b> used StandardScaler.</li>"
        "<li><b>Decision Tree, Random Forest, Gradient Boosting and XGBoost</b> were used without feature scaling.</li></ul>")

    algos = [["01", "Logistic Regression"], ["02", "Support Vector Machine (SVM)"],
             ["03", "Decision Tree"], ["04", "Random Forest"],
             ["05", "Gradient Boosting"], ["06", "XGBoost"]]
    html += part("02", "Models compared",
        "<p>The same evaluation approach was applied across all six algorithms.</p>"
        + table(["No.", "Model"], algos))

    tt = [["Logistic Regression", "0.9951", "1.0000", "0.9955", "1.0000", "1.0000"],
          ["SVM", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
          ["Decision Tree", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
          ["Random Forest", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
          ["Gradient Boosting", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
          ["XGBoost", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"]]
    html += part("03", "Test-set results",
        "<p>The held-out test set contained 46 records. The table compares training and test performance.</p>"
        + table(["Model", "Train accuracy", "Test accuracy", "Train F1", "Test F1", "Test ROC-AUC"], tt)
        + "<p class=\"note\">All six models achieved 1.00 on the test metrics shown. On this small test set, those results alone do not distinguish the models.</p>")

    html += part("04", "How the models were evaluated",
        "<ul><li><b>Classification report:</b> precision, recall, F1-score and support for each class.</li>"
        "<li><b>Confusion matrix:</b> counts correct and incorrect predictions.</li>"
        "<li><b>ROC curve and ROC-AUC:</b> show how well the model separates the two classes across thresholds.</li>"
        "<li><b>Precision–Recall curve:</b> shows the trade-off between precision and recall.</li>"
        "<li><b>StratifiedGroupKFold:</b> checks consistency across folds while keeping identical feature profiles together.</li></ul>")

    cv = [["Logistic Regression", "0.991667", "0.989474", "0.991667", "0.990339", "0.998942"],
          ["Decision Tree", "0.986667", "0.963636", "1.000000", "0.980000", "0.990476"],
          ["Random Forest", "0.981667", "0.963636", "0.991667", "0.975745", "1.000000"],
          ["SVM", "0.983333", "0.956522", "1.000000", "0.975610", "1.000000"],
          ["XGBoost", "0.975606", "0.963636", "0.982971", "0.971300", "0.990476"],
          ["Gradient Boosting", "0.948729", "0.963636", "0.946358", "0.951292", "0.982143"]]
    html += part("05", "Cross-validation comparison",
        "<p>Grouped cross-validation provides another view of model performance across multiple folds.</p>"
        + table(["Model", "CV accuracy", "CV precision", "CV recall", "CV F1-score", "CV ROC-AUC"], cv, "Logistic Regression")
        + "<p class=\"note\">The test set gave all six models perfect scores on the reported metrics. Cross-validation helps show how consistently they performed across different grouped splits.</p>")

    lrcmp = [["Accuracy", "1.0000", "0.9917"], ["F1-score", "1.0000", "0.9903"], ["ROC-AUC", "1.0000", "0.9989"]]
    html += part("06", "The selected model",
        "<h4>Logistic Regression</h4>"
        "<ul><li>Highest cross-validated accuracy: <b>0.9917</b>.</li>"
        "<li>Highest cross-validated precision: <b>0.9895</b>.</li>"
        "<li>Highest cross-validated F1-score: <b>0.9903</b>.</li></ul>"
        "<p>It was selected based on the overall comparison, not because it led every metric. Random Forest and SVM reached "
        "a cross-validated ROC-AUC of 1.00, while Decision Tree and SVM reached a recall of 1.00.</p>"
        + table(["Metric", "Test set", "Cross-validation"], lrcmp)
        + "<p class=\"note\">Cross-validation helps assess consistency; it does not guarantee performance on future businesses. ROC-AUC measures class separation, not accuracy.</p>")

    rep = [["0 (Bankruptcy)", "1.00", "1.00", "1.00", "14"],
           ["1 (Non-bankruptcy)", "1.00", "1.00", "1.00", "32"],
           ["Accuracy", "", "", "1.00", "46"],
           ["Macro average", "1.00", "1.00", "1.00", "46"],
           ["Weighted average", "1.00", "1.00", "1.00", "46"]]
    html += part("07", "A closer look at Logistic Regression",
        "<h4>Classification report</h4>"
        + table(["Class", "Precision", "Recall", "F1-score", "Support"], rep)
        + '<div class="grid g2">'
        + vis("Confusion matrix", "Correct and incorrect classifications on the test set.",
              "logistic_confusion_matrix.png", "Confusion matrix", True)
        + vis("ROC curve", "ROC-AUC on the test set: 1.00.",
              "logistic_roc_curve.png", "ROC curve", True)
        + vis("Precision–Recall curve", "Average precision on the test set: 1.00.",
              "logistic_precision_recall_curve.png", "Precision-recall curve", True)
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
