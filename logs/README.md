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

**Windows Users**: You can also use the provided batch file:
```bash
quick_start.bat
```

The GUI provides an intuitive interface for:
- Configuration management
- Data analysis and visualization
- Model training with real-time metrics
- Backtesting with performance charts
- Live trading monitoring (paper trading supported)
- Model management and comparison

### Command Line Usage

1. **Configure the system**:s
```bash
python -c "from src.utils import create_default_config; create_default_config('config/config.yaml')"
```

2. **Train a model**:
```bash
python train_enhanced.py --mode new --timesteps 100000 --config config/config.yaml
```

3. **Continue training an existing model**:
```bash
python train_enhanced.py --mode continue --model models/my_model.pt --timesteps 50000
```

4. **Run continuous training**:
```bash
python train_enhanced.py --mode continuous --config config/config.yaml
```

### Training
```bash
python train_enhanced.py --mode new --config config/config.yaml --timesteps 100000
```

### Backtesting
```bash
python src/backtest.py --config config/config.yaml --model models/best_model.pt
```

### Live Trading (Paper Trading)
```bash
python src/live_trade.py --config config/config.yaml --model models/best_model.pt --paper-trade
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