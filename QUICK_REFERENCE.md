# Sentiment Decay Analysis - Quick Reference

## 🚀 Quick Commands

```powershell
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run examples
python examples.py

# Run full pipeline
cd src
python main.py

# Run tests
pytest tests/test_basic.py -v
```

## 📝 Key Formulas

### Decay Functions

**Exponential:**
```
w_j = exp(-λ * t_j)
λ: decay rate (higher = faster decay)
```

**Half-Life:**
```
M_t = α * S_t + (1 - α) * M_{t-1}
α = 1 - exp(ln(0.5) / h)
h: half-life in days
```

**Linear:**
```
w_j = 1 - β * t_j
β: linear decay slope
```

### Sentiment Score
```
sentiment_score = P(positive) - P(negative)
Range: [-1, 1]
```

### Sharpe Ratio
```
Sharpe = sqrt(252) * mean(excess_returns) / std(returns)
```

## 🎯 Module Quick Reference

| Module | Purpose | Key Class |
|--------|---------|-----------|
| `data_loader.py` | Load & align data | `DataLoader` |
| `sentiment_extractor.py` | FinBERT sentiment | `SentimentExtractor` |
| `decay_functions.py` | Decay mechanisms | `ExponentialDecay` |
| `feature_engineering.py` | Technical indicators | `FeatureEngineer` |
| `models.py` | ML models & CV | `ModelTrainer` |
| `grid_search.py` | Parameter optimization | `DecayGridSearch` |
| `evaluation.py` | Metrics & SHAP | `ModelEvaluator` |
| `reporting.py` | Visualization | `ReportGenerator` |
| `main.py` | Pipeline orchestrator | `SentimentDecayPipeline` |

## 🔧 Configuration Snippets

```yaml
# Quick config edits in config.yaml

# Speed up testing
decay:
  exponential:
    lambda_values: [0.5, 1.0, 1.5]  # Fewer values

modeling:
  train_window: 126  # Smaller window
  rolling_splits: 3   # Fewer splits

# Use CPU instead of GPU
sentiment:
  batch_size: 8  # Smaller batches
```

## 💻 Code Snippets

### Load Data
```python
from src.data_loader import load_and_prepare_data
import yaml

config = yaml.safe_load(open('config.yaml'))
market_df, news_df, aligned_df = load_and_prepare_data(config)
```

### Extract Sentiment
```python
from src.sentiment_extractor import SentimentExtractor

extractor = SentimentExtractor(model_name="ProsusAI/finbert")
sentiment_df = extractor.extract_sentiment(aligned_df)
```

### Apply Decay
```python
from src.decay_functions import ExponentialDecay

decay = ExponentialDecay(lambda_param=1.0)
df_with_weights = decay.apply_to_dataframe(sentiment_df)
```

### Train Model
```python
from src.models import ModelTrainer

trainer = ModelTrainer(config)
results = trainer.train_with_cv(df, feature_cols, 'xgboost')
```

### Evaluate
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(config)
metrics = evaluator.evaluate_cv_results(results)
```

### Generate Report
```python
from src.reporting import generate_complete_report

generate_complete_report(
    comparison_df, grid_results, grid_summary,
    best_config, sentiment_df, output_dir='results'
)
```

## 📊 Output Files

| File | Content |
|------|---------|
| `model_comparison.png` | Baseline vs Sentiment performance |
| `grid_search_sharpe_ratio.png` | Sharpe vs decay parameters |
| `sentiment_distribution.png` | Sentiment score distribution |
| `results_comparison.csv` | Metrics table |
| `summary_report.txt` | Text summary |

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| CUDA not found | `pip install torch --index-url https://...` |
| Memory error | Reduce `batch_size` in config |
| Not enough data | Reduce `train_window` in config |
| Slow sentiment | Skip step 2: `skip_steps=[2]` |

## 📈 Parameter Tuning Guide

### Exponential Decay (λ)
- **0.2-0.5**: Slow decay, long memory
- **1.0-1.5**: Moderate decay (recommended)
- **2.0-3.0**: Fast decay, recent focus

### Half-Life (h days)
- **1-2**: Very fast decay
- **3-5**: Moderate decay (recommended)
- **7+**: Slow decay, long memory

### Linear Decay (β)
- **0.25**: Slow decay
- **0.5**: Moderate decay
- **0.75-1.0**: Fast decay

## 🎓 Research Workflow

1. **Explore** → Run `examples.py`
2. **Test** → Run with `skip_steps=[2,5]`
3. **Experiment** → Edit config, re-run
4. **Full Run** → Complete pipeline
5. **Analyze** → Review results/
6. **Iterate** → Modify modules, repeat

## 🔗 Important Paths

```
Data:
- AAPL_cleaned.csv
- MSFT_cleaned.csv
- reuters_headlines.csv

Config:
- config.yaml

Code:
- src\*.py

Results:
- results\*
```

## 📚 Documentation

- **README.md** - Full documentation
- **GETTING_STARTED.md** - Setup guide
- **PROJECT_SUMMARY.md** - Overview
- **This file** - Quick reference

## ✅ Validation Checklist

- [ ] Data files present
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list`)
- [ ] Config.yaml edited
- [ ] Examples run successfully
- [ ] Main pipeline tested
- [ ] Results generated
- [ ] Metrics reviewed

## 🎯 Expected Runtime

| Step | Time | Can Skip? |
|------|------|-----------|
| 1. Data loading | 1 min | No |
| 2. Sentiment extraction | 15-30 min | Yes |
| 3. Baseline modeling | 5-10 min | No |
| 4. Flat sentiment | 5-10 min | No |
| 5. Grid search | 30-60 min | Yes |
| 6. Optimal decay | 5-10 min | No |
| 7. Evaluation | 1 min | No |
| 8. SHAP | 5 min | Yes |
| 9. Reporting | 1 min | No |

**Total: ~1-2 hours (full), ~30 min (skipping slow steps)**

## 💡 Pro Tips

1. Test with small data first
2. Use GPU for sentiment extraction
3. Cache intermediate results
4. Start with narrow parameter grid
5. Monitor memory usage
6. Save checkpoints
7. Version control your configs
8. Document your experiments

---

**Keep this file handy while working!** 🚀
