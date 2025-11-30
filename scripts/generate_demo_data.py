"""
Script to generate pre-computed results for the dashboard demo.
This allows the dashboard to run quickly without real-time computation.

Run with: python scripts/generate_demo_data.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from utils.data_loader import get_dataset, TimeSeriesDataLoader
from utils.feature_engineering import TemporalFeatureEngineer
from utils.metrics import ForecastMetrics, ExplanationMetrics


def generate_demo_results(
    dataset_name: str = 'stock',
    model_name: str = 'xgboost',
    output_dir: str = None
):
    """
    Generate pre-computed results for a dataset/model combination.
    """
    print(f"\n{'='*60}")
    print(f"Generating demo data for {dataset_name} with {model_name}")
    print('='*60)
    
    if output_dir is None:
        output_dir = project_root / 'results' / 'precomputed'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📊 Loading data...")
    df = get_dataset(dataset_name)
    
    target_col = {
        'stock': 'close',
        'weather': 'temperature',
        'energy': 'consumption'
    }[dataset_name]
    
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target column: {target_col}")
    
    # Feature engineering
    print("\n🔧 Engineering features...")
    fe = TemporalFeatureEngineer(target_col=target_col, date_col='date')
    df_features, feature_names = fe.fit_transform(df)
    
    print(f"   Features created: {len(feature_names)}")
    
    # Temporal split
    print("\n✂️ Splitting data temporally...")
    loader = TimeSeriesDataLoader()
    train_df, val_df, test_df = loader.temporal_split(df_features)
    
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare feature matrices
    feature_cols = [col for col in feature_names if col in df_features.columns]
    
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    test_dates = pd.to_datetime(test_df['date'].values)
    
    # Train model
    print(f"\n🤖 Training {model_name} model...")
    
    from models import get_model
    model = get_model(model_name)
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    
    # Get predictions
    print("\n📈 Generating predictions...")
    predictions = model.predict(X_test)
    forecast_result = model.predict_with_confidence(X_test)
    
    # Calculate forecast metrics
    metrics = ForecastMetrics.compute_all(y_test, predictions, y_train)
    print(f"   RMSE: {metrics['RMSE']:.4f}")
    print(f"   MAE: {metrics['MAE']:.4f}")
    print(f"   R²: {metrics['R2']:.4f}")
    
    # Create explainer
    print("\n🔬 Creating hybrid explainer...")
    from explainers import create_hybrid_explainer
    
    predict_func = model.predict
    model_obj = model.model if hasattr(model, 'model') else None
    
    explainer = create_hybrid_explainer(
        model=predict_func,
        feature_names=feature_cols,
        training_data=X_train,
        model_object=model_obj if model_name == 'xgboost' else None,
        fusion_method='adaptive'
    )
    
    # Generate explanations for test set
    print("\n💡 Generating explanations (this may take a while)...")
    n_explain = min(100, len(X_test))  # Limit for storage
    
    explanations = explainer.explain_temporal_window(
        X_test[-n_explain:],
        timestamps=test_dates[-n_explain:]
    )
    
    print(f"   Generated {len(explanations)} explanations")
    
    # Extract matrices
    shap_matrix, lime_matrix, hybrid_matrix = explainer.get_temporal_importance_matrix(explanations)
    
    # Compute coherence scores
    print("\n📏 Computing coherence scores...")
    coherence_scores = explainer.compute_coherence_scores(explanations)
    print(f"   SHAP coherence: {coherence_scores['shap_coherence']:.4f}")
    print(f"   LIME coherence: {coherence_scores['lime_coherence']:.4f}")
    print(f"   Hybrid coherence: {coherence_scores['hybrid_coherence']:.4f}")
    
    # Save results
    print(f"\n💾 Saving results to {output_dir}...")
    
    prefix = f"{dataset_name}_{model_name}"
    
    # Save model
    joblib.dump(model, output_dir / f'{prefix}_model.pkl')
    
    # Save data
    np.save(output_dir / f'{prefix}_X_test.npy', X_test[-n_explain:])
    np.save(output_dir / f'{prefix}_y_test.npy', y_test[-n_explain:])
    np.save(output_dir / f'{prefix}_predictions.npy', predictions[-n_explain:])
    
    # Save confidence intervals
    np.save(output_dir / f'{prefix}_conf_lower.npy', forecast_result.confidence_lower[-n_explain:])
    np.save(output_dir / f'{prefix}_conf_upper.npy', forecast_result.confidence_upper[-n_explain:])
    
    # Save explanation matrices
    np.save(output_dir / f'{prefix}_shap_matrix.npy', shap_matrix)
    np.save(output_dir / f'{prefix}_lime_matrix.npy', lime_matrix)
    np.save(output_dir / f'{prefix}_hybrid_matrix.npy', hybrid_matrix)
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'model_name': model_name,
        'target_col': target_col,
        'feature_names': feature_cols,
        'n_explain': n_explain,
        'forecast_metrics': metrics,
        'coherence_scores': coherence_scores,
        'test_dates': test_dates[-n_explain:].tolist(),
        'generated_at': datetime.now().isoformat()
    }
    
    joblib.dump(metadata, output_dir / f'{prefix}_metadata.pkl')
    
    print(f"\n✅ Done! Results saved with prefix: {prefix}")
    
    return metadata


def generate_all_combinations():
    """Generate demo data for all dataset/model combinations."""
    datasets = ['stock', 'weather', 'energy']
    models = ['xgboost']  # Start with XGBoost as it's fastest
    
    all_results = []
    
    for dataset in datasets:
        for model in models:
            try:
                result = generate_demo_results(dataset, model)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Failed for {dataset}/{model}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Generated {len(all_results)} demo datasets")
    print('='*60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate demo data for the dashboard")
    parser.add_argument('--dataset', type=str, default='all', 
                       choices=['stock', 'weather', 'energy', 'all'])
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'prophet', 'lstm'])
    
    args = parser.parse_args()
    
    if args.dataset == 'all':
        generate_all_combinations()
    else:
        generate_demo_results(args.dataset, args.model)
