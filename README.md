# Jane Street Real-Time Market Data Forecasting

GRU ensemble with daily online learning for the [Jane Street Kaggle competition](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting). Architecture matches the #8 Private LB solution (score: 0.013 vs. LightGBM baseline: 0.007).

---

## Architecture

```
79 raw features
  → per-symbol rolling mean + std (1,000-step window)
  → 237 engineered features
  → Linear input projection + LayerNorm
  → 2-layer GRU (hidden size 700)
  → MLP head → tanh × 5
  → prediction ∈ [-5, 5]
```

**Ensemble:** 6 models trained with different random seeds, averaged at inference.

---

## Key Design Decisions

**Stateful inference**
Each symbol maintains its own GRU hidden state across intraday batches. Hidden states persist within a trading session and reset at day boundaries (`time_id=0`), preserving within-session temporal memory without cross-day contamination.

**Daily online learning**
At the start of each new day, all 6 models are fine-tuned for 3 gradient steps on the previous day's realized `responder_6` values from the competition lags API. This enables continuous adaptation to non-stationary market regimes — the core limitation of static offline models.

**Rolling feature engineering**
Raw features alone carry weak signal. Per-symbol rolling statistics over a 1,000-step window capture intraday momentum and volatility dynamics. This is computed incrementally at inference using a per-symbol deque buffer, with no lookahead.

---

## Results

| Model | Private LB (weighted R²) |
|---|---|
| LightGBM baseline | 0.007 |
| GRU ensemble (this architecture) | 0.013 |

---

## Requirements

```
torch>=2.0
polars>=0.20
numpy
pandas
kaggle_evaluation
```

---

## Usage

**Training (requires ~100GB RAM + GPU):**
```bash
# Run all cells in jane_street_gru.ipynb
# Set INPUT_DIR to your local data path
```

**Inference on Kaggle:**
```python
# The final cell launches the JSInferenceServer
# Kaggle will call predict(test, lags) for each batch
inference_server = JSInferenceServer(predict)
inference_server.serve()
```
