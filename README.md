# AI PPO Trading Bot

A reinforcement learning trading bot that uses Proximal Policy Optimization (PPO) to learn trading strategies from TradingView API data.

## Features

- PPO-based reinforcement learning for trading
- TradingView API integration for real-time market data
- Comprehensive backtesting and performance analysis
- Technical indicator integration
- Risk management and portfolio optimization
- Real-time trading visualization

## Project Structure

```
ai-ppo/
├── src/                    # Source code
│   ├── agents/            # PPO agent implementation
│   ├── environments/      # Trading environment
│   ├── data/             # Data fetching and processing
│   ├── utils/            # Utility functions
│   └── visualization/    # Plotting and analysis tools
├── config/               # Configuration files
├── data/                # Historical data storage
├── models/              # Trained model checkpoints
├── logs/                # Training logs and metrics
└── tests/               # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MasstarVT/ai-ppo.git
cd ai-ppo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp config/config_template.yaml config/config.yaml
# Edit config.yaml with your API credentials and preferences
```

## Usage

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