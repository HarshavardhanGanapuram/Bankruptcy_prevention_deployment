# Bankruptcy Prevention

A machine learning project that classifies a business as **Bankruptcy** or **Non-bankruptcy** using six qualitative business factors.

The deployed application uses a saved scikit-learn Pipeline containing `StandardScaler` and `LogisticRegression`, stored as `bankruptcy_model.pkl`.

## Project Overview

The project covers the complete workflow:

- Data exploration and preprocessing
- Exploratory data analysis and visualisation
- Training and evaluation of six classification models
- Cross-validation and model comparison
- Selection of Logistic Regression as the final model
- Deployment through a Streamlit application

## Repository Structure

```text
Bankruptcy_prevention_deployment/
│
├── assets/
│   ├── bank_building.png
│   ├── exploration.jpg
│   ├── evaluation.jpg
│   ├── prediction.jpg
│   └── charts/
│
├── Bankruptcy_prevention_app.py
├── Bankruptcy_Prevention_Notebook.ipynb
├── bankruptcy_model.pkl
├── README.md
├── requirements.txt
└── config.toml
```

The notebook is included for the complete analysis and model-building workflow. The Streamlit application uses the Python app file and the saved model; it does not execute the notebook.

> Keep the filenames and folder paths in this outline aligned with the files actually uploaded to the repository. If `config.toml` is intended to configure Streamlit's theme, Streamlit expects it at `.streamlit/config.toml`.

## Run Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Bankruptcy_prevention_deployment
```

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run Bankruptcy_prevention_app.py
```

## Files

| File / Folder | Purpose |
|---|---|
| `Bankruptcy_prevention_app.py` | Streamlit application for interactive bankruptcy prediction |
| `Bankruptcy_Prevention_Notebook.ipynb` | Jupyter Notebook containing data exploration, preprocessing, model training and evaluation |
| `bankruptcy_model.pkl` | Saved Logistic Regression pipeline used by the application |
| `assets/` | Images and visual assets used in the application |
| `assets/charts/` | EDA and model-evaluation charts, if included |
| `requirements.txt` | Python package dependencies |
| `config.toml` | Configuration file currently included in the repository |

## Model Information

Six classification algorithms were evaluated:

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost

Based on the cross-validation comparison, **Logistic Regression** was selected as the final model because it achieved the strongest overall balance across accuracy, precision and F1-score.

The final saved pipeline combines:

- `StandardScaler` for feature scaling
- `LogisticRegression` for classification

The application loads `bankruptcy_model.pkl` to make predictions. Keep this file in the repository alongside the Streamlit application, and use compatible scikit-learn versions when creating and loading the saved model.

## Deployment

The application is deployed using **Streamlit Community Cloud** and connected to this GitHub repository.

To deploy your own copy:

1. Push the project files to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app.
3. Select the repository and branch.
4. Set the main file path to `Bankruptcy_prevention_app.py`.
5. Deploy the app.

Ensure that `bankruptcy_model.pkl`, `requirements.txt`, and the required `assets/` files are present in the repository.

## Notes

- The Jupyter Notebook is provided for reviewing the analysis and model-development process.
- Streamlit runs `Bankruptcy_prevention_app.py`; it does not run the notebook.
- The saved model file must correspond to the model pipeline expected by the application.
