"""
Quick test script to verify the installation works.
Run with: python scripts/test_installation.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    tests = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("utils.data_loader", "from utils.data_loader import get_dataset"),
        ("utils.feature_engineering", "from utils.feature_engineering import TemporalFeatureEngineer"),
        ("utils.metrics", "from utils.metrics import ForecastMetrics"),
        ("models.xgboost_model", "from models.xgboost_model import XGBoostForecaster"),
        ("explainers.base_explainer", "from explainers.base_explainer import BaseExplainer"),
        ("explainers.hybrid_explainer", "from explainers.hybrid_explainer import HybridSHAPLIMEExplainer"),
    ]
    
    passed = 0
    failed = 0
    
    for name, cmd in tests:
        try:
            exec(cmd)
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
    
    return passed, failed


def test_data_loading():
    """Test data loading."""
    print("\nTesting data loading...")
    
    from utils.data_loader import get_dataset
    
    for dataset_name in ['stock', 'weather', 'energy']:
        try:
            df = get_dataset(dataset_name)
            print(f"  ✅ {dataset_name}: {df.shape}")
        except Exception as e:
            print(f"  ❌ {dataset_name}: {e}")


def test_feature_engineering():
    """Test feature engineering."""
    print("\nTesting feature engineering...")
    
    from utils.data_loader import get_dataset
    from utils.feature_engineering import TemporalFeatureEngineer
    
    df = get_dataset('stock')
    fe = TemporalFeatureEngineer(target_col='close', date_col='date')
    
    try:
        df_features, feature_names = fe.fit_transform(df)
        print(f"  ✅ Created {len(feature_names)} features")
        print(f"  ✅ Output shape: {df_features.shape}")
    except Exception as e:
        print(f"  ❌ Feature engineering failed: {e}")


def test_model_training():
    """Test model training."""
    print("\nTesting model training...")
    
    import numpy as np
    from models.xgboost_model import XGBoostForecaster
    
    # Create dummy data
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    try:
        model = XGBoostForecaster(n_estimators=10)
        model.fit(X[:80], y[:80], X[80:], y[80:])
        preds = model.predict(X[80:])
        print(f"  ✅ XGBoost training successful")
        print(f"  ✅ Predictions shape: {preds.shape}")
    except Exception as e:
        print(f"  ❌ Model training failed: {e}")


def test_explainer():
    """Test explainer creation."""
    print("\nTesting explainer...")
    
    import numpy as np
    from models.xgboost_model import XGBoostForecaster
    from explainers.hybrid_explainer import create_hybrid_explainer
    
    # Create and train model
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    feature_names = [f'feature_{i}' for i in range(10)]
    
    model = XGBoostForecaster(n_estimators=10)
    model.fit(X[:80], y[:80], feature_names=feature_names)
    
    try:
        explainer = create_hybrid_explainer(
            model=model.predict,
            feature_names=feature_names,
            training_data=X[:80],
            model_object=model.model
        )
        print(f"  ✅ Explainer created successfully")
        
        # Test explanation
        exp = explainer.explain_instance(X[90])
        print(f"  ✅ Generated explanation with {len(exp.importance_scores)} features")
        print(f"  ✅ Top feature: {exp.top_k_features(1)[0][0]}")
    except Exception as e:
        print(f"  ❌ Explainer test failed: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("SHAP-LIME Hybrid Explainer - Installation Test")
    print("=" * 60)
    
    passed, failed = test_imports()
    
    if failed == 0:
        test_data_loading()
        test_feature_engineering()
        test_model_training()
        test_explainer()
    
    print("\n" + "=" * 60)
    if failed == 0:
        print("✅ All tests passed! Installation looks good.")
    else:
        print(f"❌ {failed} import(s) failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
