# Sentiment Decay Analysis for Stock Market Prediction

A comprehensive research framework for studying how **sentiment decay mechanisms** influence the accuracy of predictive stock market models.

## 📋 Overview

This project implements a complete pipeline for:
- Extracting sentiment from financial news using FinBERT
- Applying various decay functions (exponential, half-life, linear)
- Training predictive models with technical indicators and sentiment features
- Optimizing decay parameters through grid search
- Comprehensive evaluation with trading metrics and SHAP analysis

## 🏗️ Project Structure

```
Thesis/
├── config.yaml                 # Configuration file
├── requirements.txt           # Python dependencies
├── src/                       # Source code modules
│   ├── data_loader.py        # Data loading and alignment
│   ├── sentiment_extractor.py # FinBERT sentiment extraction
│   ├── decay_functions.py    # Decay mechanisms
│   ├── feature_engineering.py # Technical indicators
│   ├── models.py             # ML models and CV
│   ├── grid_search.py        # Parameter optimization
│   ├── evaluation.py         # Metrics and SHAP
│   ├── reporting.py          # Visualization and reports
│   └── main.py               # Main orchestrator
├── results/                   # Output directory
├── notebooks/                 # Jupyter notebooks
└── tests/                     # Unit tests
```

## 📊 Data Requirements

The project expects three CSV files in the root directory:

1. **AAPL_cleaned.csv** / **MSFT_cleaned.csv**
   - Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
   
2. **reuters_headlines.csv**
   - Columns: `Headlines`, `Time`, `Description`

## 🚀 Installation

### 1. Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. (Optional) Install PyTorch with CUDA for GPU acceleration

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Usage

### Quick Start

Run the complete pipeline:

```powershell
cd src
python main.py
```

**⚡ Note:** First run takes 15-30 minutes (FinBERT sentiment extraction). Subsequent runs are **20-30x faster** thanks to automatic caching! See [CACHING_GUIDE.md](CACHING_GUIDE.md) for details.

### Configuration

Edit `config.yaml` to customize:
- Data paths and cache settings
- Decay parameters to test
- Model hyperparameters
- Rolling window sizes
- Evaluation metrics

See [PARAMETER_TUNING_GUIDE.md](PARAMETER_TUNING_GUIDE.md) for optimization recommendations.

### Step-by-Step Execution

```python
from src.main import SentimentDecayPipeline

# Initialize pipeline
pipeline = SentimentDecayPipeline(config_path="config.yaml")

# Run specific steps
pipeline.run_data_loading()          # Step 1: Load data
pipeline.run_sentiment_extraction()  # Step 2: Extract sentiment (cached after first run!)
pipeline.run_baseline_modeling()     # Step 3: Baseline models
pipeline.run_flat_sentiment_modeling()  # Step 4: Flat sentiment
pipeline.run_grid_search()           # Step 5: Grid search
pipeline.run_optimal_decay_modeling('exponential', 1.0)  # Step 6
pipeline.run_evaluation()            # Step 7: Evaluate
pipeline.generate_report()           # Step 9: Generate report

# Or run full pipeline
pipeline.run_full_pipeline()
```

### Cache Management

```powershell
# View cached sentiment extractions
python cache_manager.py list

# Clear cache to force recomputation
python cache_manager.py clear

# Inspect cache contents
python cache_manager.py inspect --file sentiment_XXXXXXXX.pkl

# See full documentation
# Read CACHING_GUIDE.md
```

## 🧪 Research Workflow

### 1. Data Preparation
- Load OHLCV market data
- Load news headlines
- Align news with trading days
- Extract ticker mentions from headlines

### 2. Sentiment Analysis
- Use FinBERT (`ProsusAI/finbert`) for sentiment scoring
- Score = P(positive) - P(negative)
- Aggregate daily sentiment statistics

### 3. Decay Functions

**Exponential Intraday Decay:**
```
w_j = exp(-λ * t_j)
```

**Half-Life Cross-Day Decay:**
```
M_t = α * S_t + (1 - α) * M_{t-1}
α = 1 - exp(ln(0.5) / h)
```

**Linear Decay:**
```
w_j = 1 - β * t_j
```

### 4. Feature Engineering
- Technical indicators: RSI, MACD, EMA, Momentum, Volatility
- Flat sentiment features: mean, std, min, max, count
- Decay sentiment features: weighted mean, variance, memory

### 5. Modeling
- Models: Logistic Regression, XGBoost
- Cross-validation: Rolling time-series split (252-day train, 21-day test)
- Target: Binary classification (Close[t+1] > Close[t])

### 6. Grid Search
- Test multiple decay parameters
- Optimize for Sharpe ratio or AUC
- Compare across decay types

### 7. Evaluation
- Classification metrics: Accuracy, F1, AUC
- Trading metrics: Sharpe ratio, max drawdown, cumulative returns
- SHAP analysis for feature importance

### 8. Reporting
- Comparative tables (baseline vs flat vs decay)
- Visualizations (AUC vs λ, Sharpe vs λ)
- Feature importance rankings
- Summary PDF/notebook

## 📈 Expected Outcomes

1. **Quantitative proof** that decay-weighted sentiment outperforms static sentiment
2. **Empirical relationship** between decay rate and market efficiency
3. **Optimal decay parameters** for short-term prediction (expected λ ≈ 1-1.5)
4. Foundation for **adaptive-decay** or **regime-aware** modeling

## 🔬 Extending the Research

### Add new decay functions
```python
# src/decay_functions.py
class CustomDecay(DecayFunction):
    def compute_weights(self, normalized_times):
        # Your custom decay logic
        return weights
```

### Add new features
```python
# src/feature_engineering.py
class TechnicalIndicators:
    @staticmethod
    def add_custom_indicator(df):
        # Your custom indicator
        return df
```

### Add new models
```python
# src/models.py
from sklearn.ensemble import RandomForestClassifier

class BaseModel:
    def _initialize_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.kwargs)
```

## 🐛 Troubleshooting

### Issue: ImportError for torch/transformers
```powershell
pip install torch transformers
```

### Issue: ImportError for ta
```powershell
pip install ta
```

### Issue: Memory errors during sentiment extraction
- Reduce `batch_size` in config.yaml
- Process fewer headlines
- Use CPU instead of GPU (slower but uses less memory)

### Issue: Not enough data for rolling CV
- Reduce `train_window` or `test_window` in config.yaml
- Reduce `rolling_splits`

### Issue: Sentiment extraction is slow
- **Use caching!** Second run will be 20-30x faster
- Enable GPU acceleration (install PyTorch with CUDA)
- Reduce batch size if GPU memory limited
- See [CACHING_GUIDE.md](CACHING_GUIDE.md) for performance tips

### Issue: Cache not loading
```powershell
# Check cache status
python cache_manager.py info

# Clear corrupted cache
python cache_manager.py clear --yes
```

## 📚 Documentation

- **[README.md](README.md)** - Main documentation (you are here)
- **[CACHING_GUIDE.md](CACHING_GUIDE.md)** - Caching system and performance optimization
- **[PARAMETER_TUNING_GUIDE.md](PARAMETER_TUNING_GUIDE.md)** - Hyperparameter optimization guide
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - High-level project overview
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference for common tasks

## 📝 Citation

If you use this code in your research, please cite:

```
@software{sentiment_decay_analysis,
  title = {Sentiment Decay Analysis for Stock Market Prediction},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/sentiment-decay}
}
```

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/repo/issues)

## 🙏 Acknowledgments

- FinBERT model: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- Technical indicators: [ta library](https://github.com/bukosabino/ta)
- SHAP: [shap library](https://github.com/slundberg/shap)
