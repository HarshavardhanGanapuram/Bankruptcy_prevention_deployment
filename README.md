# Bankruptcy Prevention

A Streamlit app that classifies a business as bankruptcy or non-bankruptcy
from six qualitative factors. The model is a scikit-learn Pipeline
(StandardScaler + LogisticRegression) saved as `bankruptcy_model.pkl`.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run Bankruptcy_prevention_app.py
```

## Files

- `Bankruptcy_prevention_app.py` the app
- `bankruptcy_model.pkl` your saved Pipeline (place it next to the app)
- `assets/` illustrations (`bank_building.png`, `exploration.jpg`, `evaluation.jpg`, `prediction.jpg`) and `assets/charts/` (evaluation and EDA charts). Commit this whole folder.
- `backup/` the previous version of the app
- `.streamlit/config.toml` theme colours
- `requirements.txt`

## Deploy

Push the folder to GitHub, then create an app on Streamlit Community Cloud
and point it at `Bankruptcy_prevention_app.py`. Keep the `.pkl` in the repo.
Train the model with the same scikit-learn version listed in requirements
(or pin it) so the pickle loads cleanly.
