# GUI Application

This directory contains the web-based GUI for the AI PPO Trading System built with Streamlit.

## Features

The GUI provides the following features:

### 📈 Dashboard
- Portfolio performance overview
- Key metrics (returns, Sharpe ratio, drawdown)
- Performance charts and heatmaps
- Recent trading activity

### ⚙️ Configuration
- Trading parameters (symbols, timeframe, balance)
- PPO hyperparameters
- Risk management settings
- Data source configuration

### 📊 Data Analysis
- Symbol selection and date range
- Price charts and technical analysis
- Volume and volatility metrics
- Market data visualization

### 🎯 Training
- Model training interface
- Hyperparameter configuration
- Real-time training metrics
- Training progress monitoring

### 📊 Backtesting
- Model selection for backtesting
- Backtest parameter configuration
- Performance comparison with benchmarks
- Detailed results and metrics

### 📡 Live Trading
- Real-time portfolio monitoring
- Position management
- Market data feeds
- Paper trading mode

### 📁 Model Management
- View trained models
- Model comparison tools
- Export/import functionality
- Model analysis

## Running the GUI

### Prerequisites

Make sure you have installed all required packages:

```bash
pip install -r requirements.txt
```

### Starting the Application

From the project root directory:

```bash
cd gui
python run_gui.py
```

Or directly with Streamlit:

```bash
streamlit run gui/app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Configuration

The GUI integrates with the configuration system in `config/config.yaml`. You can:

- Modify settings through the Configuration page
- Save/load configurations
- Reset to default settings

## Architecture

### Files

- `app.py` - Main Streamlit application
- `run_gui.py` - Entry point script
- `README.md` - This documentation

### Components

The GUI is organized into pages using Streamlit's navigation:

1. **Dashboard** - Overview and key metrics
2. **Configuration** - System settings and parameters
3. **Data Analysis** - Market data visualization
4. **Training** - Model training interface
5. **Backtesting** - Strategy evaluation
6. **Live Trading** - Real-time trading monitor
7. **Model Management** - Model operations

### Integration

The GUI integrates with the core trading system modules:

- `src.data` - Data client and preprocessing
- `src.environments` - Trading environment
- `src.agents` - PPO agent
- `src.evaluation` - Backtesting system
- `src.utils` - Configuration and utilities

## Features in Detail

### Dashboard
- Real-time portfolio metrics
- Performance visualization
- Monthly returns heatmap
- Recent trading activity

### Configuration Management
- Trading parameters (symbols, balance, fees)
- PPO hyperparameters (learning rate, batch size, etc.)
- Risk management (drawdown limits, leverage)
- Data provider settings

### Training Interface
- Start/stop/pause training
- Real-time metrics display
- Progress visualization
- Model configuration

### Backtesting
- Model selection
- Historical performance analysis
- Benchmark comparison
- Detailed metrics and charts

### Live Trading Monitor
- Real-time portfolio status
- Position management
- Market data feeds
- Manual trading controls
- Paper trading support

## Safety Features

- Paper trading mode by default
- Configuration validation
- Risk management alerts
- Clear warnings for live trading

## Customization

The GUI can be extended by:

1. Adding new pages to the navigation menu
2. Creating additional chart types
3. Integrating new data sources
4. Adding custom metrics and indicators

## Troubleshooting

### Import Errors
If you see import errors, ensure:
- All dependencies are installed: `pip install -r requirements.txt`
- The Python path includes the `src` directory
- You're running from the correct directory

### Port Issues
If port 8501 is in use, Streamlit will automatically try other ports or you can specify:
```bash
streamlit run app.py --server.port 8502
```

### Configuration Issues
If configuration doesn't load:
- Check that `config/config.yaml` exists
- Verify YAML syntax
- Use the "Reset to Defaults" button if needed

## Development

To modify the GUI:

1. Edit `app.py` for main functionality
2. Add new pages by creating functions and updating the navigation
3. Modify styling in the CSS section
4. Test with sample data before connecting to live systems

The GUI uses session state to maintain data across page refreshes and user interactions.