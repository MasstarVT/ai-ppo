# AI PPO Trading System

A sophisticated Proximal Policy Optimization (PPO) reinforcement learning system for automated stock trading. This system learns to trade stocks using historical market data and technical indicators.

## Features

- **Reinforcement Learning**: Uses PPO algorithm for learning optimal trading strategies
- **Data Integration**: Supports multiple data sources (Yahoo Finance, Alpha Vantage, Polygon)
- **Technical Indicators**: 20+ built-in technical indicators (SMA, EMA, RSI, MACD, etc.)
- **Risk Management**: Built-in position sizing and risk controls
- **Backtesting**: Comprehensive backtesting with walk-forward analysis
- **Visualization**: Rich plotting and analysis tools
- **Configuration**: Flexible YAML/JSON configuration system
- **Web GUI**: Modern Streamlit-based dashboard for training, backtesting, and monitoring

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MasstarVT/ai-ppo.git
cd ai-ppo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run setup to verify installation:
```bash
python setup.py
```

### Web GUI (Recommended)

Launch the web-based dashboard:
```bash
cd gui
python run_gui.py
```

Or directly with Streamlit:
```bash
streamlit run gui/app.py
```

The GUI provides an intuitive interface for:
- Configuration management
- Data analysis and visualization
- Model training with real-time metrics
- Backtesting with performance charts
- Live trading monitoring (paper trading supported)
- Model management and comparison

### Command Line Usage

1. **Configure the system**:
```bash
python -c "from src.utils import create_default_config; create_default_config('config/config.yaml')"
```

2. **Run a demo**:
```bash
python demo.py
```

3. **Train a model**:
```bash
python src/train.py
```

4. **Run backtesting**:
```bash
python src/backtest.py
```

### Training
```bash
python src/train.py --config config/config.yaml
```

### Backtesting
```bash
python src/backtest.py --config config/config.yaml --model models/best_model.zip
```

### Live Trading (Paper Trading)
```bash
python src/live_trade.py --config config/config.yaml --model models/best_model.zip --paper-trade
```

## Configuration

Edit `config/config.yaml` to customize:
- Trading parameters (symbols, timeframes, position sizing)
- PPO hyperparameters (learning rate, batch size, etc.)
- Risk management settings
- API credentials

## Warning

This is for educational and research purposes only. Trading involves substantial risk of loss. Always use paper trading first and never risk more than you can afford to lose.

## License

MIT License