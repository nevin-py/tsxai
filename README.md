# 🎯 Adaptive SHAP-LIME Hybrid Explainer for Time Series Forecasting

A novel explainability framework that combines SHAP and LIME for time-series models **without temporal data leakage**.

## 🌟 Key Features

- **Temporal-Aware Perturbation**: Custom perturbation strategies that respect causal ordering
- **SHAP-LIME Fusion**: Synchronized explanations combining global and local interpretability
- **Multi-Model Support**: XGBoost, Prophet, and LSTM models
- **Interactive Dashboard**: Streamlit-based visualization with real-time explanations
- **Counterfactual Analysis**: "What-if" scenarios for temporal features

## 🏗️ Project Structure

```
shap-lime/
├── app/
│   ├── streamlit_app.py      # Main Streamlit dashboard
│   └── components/           # UI components
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Processed features
├── explainers/
│   ├── base_explainer.py     # Base explainer class
│   ├── temporal_shap.py      # Temporal-aware SHAP
│   ├── temporal_lime.py      # Temporal-aware LIME
│   └── hybrid_explainer.py   # SHAP-LIME fusion
├── models/
│   ├── base_model.py         # Base model interface
│   ├── xgboost_model.py      # XGBoost forecaster
│   ├── prophet_model.py      # Prophet forecaster
│   └── lstm_model.py         # LSTM forecaster
├── utils/
│   ├── data_loader.py        # Data loading utilities
│   ├── feature_engineering.py # Temporal feature creation
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting utilities
├── notebooks/
│   └── exploration.ipynb     # Data exploration
├── results/
│   └── precomputed/          # Pre-computed explanations
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

## 📊 Datasets

1. **Stock Prices**: S&P 500 historical data
2. **Weather Data**: Temperature and precipitation forecasting
3. **Energy Consumption**: Hourly energy demand

## 🔬 Technical Innovation

### The Temporal Leakage Problem

Traditional SHAP/LIME perturbation strategies can accidentally use future information when explaining time series predictions:

```
Standard LIME: Perturbs features randomly → May create impossible temporal sequences
Our Approach: Constrained perturbation → Respects causal ordering
```

### Hybrid Fusion Algorithm

```
1. Compute SHAP values with temporal masking
2. Compute LIME coefficients with rolling window perturbation
3. Align explanations using temporal coherence weighting
4. Output: Fused importance scores with confidence intervals
```

## 📈 Performance Metrics

- **Forecast Accuracy**: RMSE, MAE, MAPE
- **Explanation Faithfulness**: Correlation with model behavior
- **Temporal Coherence**: Explanation stability across time
- **Computation Time**: Efficiency comparison

## 🎓 Academic References

- Ribeiro et al. (2016) - LIME
- Lundberg & Lee (2017) - SHAP
- Temporal XAI Survey (2024)

## 📜 License

MIT License
