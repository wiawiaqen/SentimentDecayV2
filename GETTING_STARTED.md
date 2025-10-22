# Getting Started with Sentiment Decay Analysis

## Quick Start Guide

### 1. Prerequisites

Ensure you have Python 3.8+ installed:
```powershell
python --version
```

### 2. Setup Virtual Environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Verify activation (you should see (venv) in your prompt)
```

### 3. Install Dependencies

```powershell
# Install all required packages
pip install -r requirements.txt

# This installs:
# - pandas, numpy, scipy (data manipulation)
# - scikit-learn, xgboost (machine learning)
# - transformers, torch (sentiment analysis with FinBERT)
# - ta (technical indicators)
# - shap (explainability)
# - matplotlib, seaborn, plotly (visualization)
# - pyyaml (configuration)
```

### 4. Prepare Your Data

Place these files in the root directory:
- `AAPL_cleaned.csv` (Date, Open, High, Low, Close, Volume)
- `MSFT_cleaned.csv` (Date, Open, High, Low, Close, Volume)
- `reuters_headlines.csv` (Headlines, Time, Description)

### 5. Configure the Pipeline

Edit `config.yaml` to set:
- Data file paths
- Decay parameters to test
- Model hyperparameters
- Output directory

### 6. Run Examples First (Recommended)

Test individual components:
```powershell
python examples.py
```

This will demonstrate:
- ✓ Data loading
- ✓ Technical indicators
- ✓ Decay functions
- ✓ Model training
- ✓ Evaluation metrics

### 7. Run Full Pipeline

#### Option A: Run Everything
```powershell
cd src
python main.py
```

#### Option B: Skip Sentiment Extraction (Faster Testing)
```python
# Edit src/main.py and modify:
pipeline.run_full_pipeline(
    skip_steps=[2],  # Skip sentiment extraction
    optimal_decay_type='exponential',
    optimal_param=1.0
)
```

Then run:
```powershell
cd src
python main.py
```

### 8. View Results

After completion, check the `results/` directory:
- `model_comparison.png` - Performance comparison chart
- `grid_search_*.png` - Parameter optimization plots
- `sentiment_distribution.png` - Sentiment analysis
- `summary_report.txt` - Text summary
- `results_comparison.csv` - Detailed metrics
- `results_grid_search.csv` - Grid search results

## Pipeline Steps Explained

### Step 1: Data Loading (Fast)
- Loads OHLCV data for AAPL and MSFT
- Loads Reuters headlines
- Aligns news with trading days
- Creates binary target (price up/down)

### Step 2: Sentiment Extraction (Slow - GPU recommended)
- Uses FinBERT to score each headline
- Sentiment score = P(positive) - P(negative)
- Aggregates daily sentiment statistics
- **Note:** This step can take 10-30 minutes depending on data size

### Step 3: Baseline Modeling (Medium)
- Computes technical indicators (RSI, MACD, EMA, etc.)
- Trains models with technical features only
- Establishes baseline performance

### Step 4: Flat Sentiment Modeling (Medium)
- Adds static sentiment features
- Trains models with tech + flat sentiment
- No decay applied yet

### Step 5: Grid Search (Slow)
- Tests multiple decay parameters
- Exponential: λ = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
- Half-life: h = [1, 2, 3, 5, 7] days
- Linear: β = [0.25, 0.5, 0.75, 1.0]
- **Note:** Can take 1-2 hours for full grid

### Step 6: Optimal Decay Modeling (Fast)
- Trains model with best decay configuration
- Uses parameters from grid search

### Step 7: Evaluation (Fast)
- Computes classification metrics (Accuracy, F1, AUC)
- Computes trading metrics (Sharpe, returns, drawdown)
- Generates comparison tables

### Step 8: SHAP Analysis (Medium)
- Computes feature importance
- Shows impact of decay vs non-decay features

### Step 9: Report Generation (Fast)
- Creates visualizations
- Exports CSV files
- Generates summary text

## Troubleshooting

### Import Errors

The import errors shown are expected if packages aren't installed yet. Install with:
```powershell
pip install -r requirements.txt
```

### Memory Issues

If you get memory errors during sentiment extraction:
1. Reduce `batch_size` in config.yaml (try 8 or 16)
2. Process fewer headlines
3. Use CPU instead of GPU (slower but less memory)

### CUDA/GPU Issues

If PyTorch doesn't detect your GPU:
```powershell
# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Not Enough Data for CV

If you get errors about insufficient data:
1. Reduce `train_window` in config.yaml (try 126 = 6 months)
2. Reduce `test_window` (try 10)
3. Reduce `rolling_splits` (try 5)

## Performance Tips

### Speed Up Development
- Skip Step 2 (sentiment extraction) during testing
- Use pre-computed sentiment scores
- Reduce grid search parameters
- Test with single ticker first

### Optimize Production
- Use GPU for sentiment extraction
- Parallelize grid search with `n_jobs=-1`
- Cache intermediate results
- Use smaller SHAP sample size

## Next Steps

1. **Run examples.py** to verify installation
2. **Run main.py with skip_steps=[2]** for quick test
3. **Run full pipeline** once everything works
4. **Analyze results** in the results/ directory
5. **Customize** decay functions or features for your research

## Need Help?

- Check README.md for detailed documentation
- Review examples.py for code samples
- Check tests/test_basic.py for unit tests
- Open an issue on GitHub

## Research Extensions

Once the basic pipeline works:
- Add new decay functions in `decay_functions.py`
- Test different technical indicators in `feature_engineering.py`
- Try different ML models in `models.py`
- Implement regime detection
- Add more tickers
- Test different time horizons
- Combine multiple news sources

Happy researching! 🚀
