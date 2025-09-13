# Continuous Training Mode - User Guide

## ♾️ **Continuous Training Feature**

The continuous training mode allows you to train models indefinitely until you manually stop them. This is perfect for achieving your 50%+ return target by allowing the model to train as long as needed.

## 🚀 **How to Use Continuous Training**

### **Option 1: Command Line Interface**

```bash
# Start continuous training from scratch
python train_enhanced.py --mode continuous

# Start from existing model
python train_enhanced.py --mode continuous --model models/your_model.pt

# Customize save intervals
python train_enhanced.py --mode continuous --save-interval 25000 --checkpoint-interval 5000

# With custom configuration
python train_enhanced.py --mode continuous --config config/high_performance_config.yaml
```

### **Option 2: GUI Interface**

1. **Launch GUI**: Run `python gui\run_gui.py`
2. **Navigate to Training Tab**
3. **Select "♾️ Continuous Training"**
4. **Choose starting point**:
   - 🆕 Fresh Model - Start from scratch
   - 📂 Existing Model - Continue from existing model
5. **Configure intervals**:
   - Save Interval: How often to save models (default: 50,000 timesteps)
   - Checkpoint Interval: How often to create backups (default: 10,000 timesteps)
6. **Click "♾️ Start Continuous Training"**

## 🛑 **How to Stop Continuous Training**

### **Multiple Stop Methods Available:**

1. **GUI Stop Button**: Click "🛑 Stop Continuous Training" in the GUI
2. **Keyboard Interrupt**: Press `Ctrl+C` in the terminal
3. **Stop File**: Create a file named `stop_training.txt` in the project directory
4. **Process Termination**: Kill the Python process

### **Graceful vs Forced Stop:**
- **Graceful**: Using stop file or GUI button - saves final model
- **Forced**: Ctrl+C or process kill - may lose current progress

## 📊 **Training Progress Monitoring**

### **Automatic Saves:**
- **Regular Models**: Saved every N timesteps (configurable)
- **Checkpoints**: Backup saves for recovery
- **Final Model**: Saved when training stops

### **Model Files Created:**
- `continuous_model_YYYYMMDD_HHMMSS.pt` - Regular saves
- `checkpoint_continuous.pt` - Latest checkpoint (overwritten)
- `continuous_final_YYYYMMDD_HHMMSS.pt` - Final model when stopped

### **Live Monitoring:**
```
🚀 Starting continuous training...
📈 Training progress (continuous mode):
  Iteration    1 | Batch timesteps: 1000 | Total: 1,000
  Iteration    2 | Batch timesteps: 1000 | Total: 2,000
  ...
  Iteration   25 | Batch timesteps: 1000 | Total: 25,000
💾 Checkpoint saved: models/checkpoint_continuous.pt
```

## 🎯 **Achieving 50%+ Returns with Continuous Training**

### **Recommended Strategy:**

1. **Start with High-Performance Config**:
   ```bash
   python train_enhanced.py --mode continuous --config config/high_performance_config.yaml
   ```

2. **Monitor Performance Regularly**:
   - Check saved models every 50K timesteps
   - Use backtesting to evaluate returns
   - Stop when 50%+ return achieved consistently

3. **Let it Run Overnight**:
   - Continuous training can run for hours/days
   - Automatic saves protect against crashes
   - Resume from checkpoints if needed

### **Performance Indicators:**
- **Target**: 50%+ total return in backtesting
- **Consistency**: Multiple evaluations showing >50% return
- **Risk Control**: Maximum drawdown <20%
- **Stability**: No overfitting or performance degradation

## 🔧 **Advanced Configuration**

### **Command Line Parameters:**
```bash
--mode continuous                    # Enable continuous training
--model PATH                        # Optional: start from existing model
--save-interval N                   # Save every N timesteps (default: 50000)
--checkpoint-interval N             # Checkpoint every N timesteps (default: 10000)
--config PATH                       # Use custom configuration file
```

### **Configuration File Options:**
```yaml
training:
  total_timesteps: 2000000          # Ignored in continuous mode
ppo:
  learning_rate: 0.0005             # Higher for faster convergence
  batch_size: 128                   # Larger for stability
network:
  policy_layers: [512, 512, 256]    # Larger for complex patterns
```

## 📈 **Monitoring Training Quality**

### **Signs of Good Training:**
- ✅ Steady improvement in rewards
- ✅ Consistent positive returns in backtesting
- ✅ Low training loss
- ✅ Stable learning curve

### **Signs to Stop Training:**
- ✅ Achieved 50%+ return consistently
- ✅ Performance plateau (no improvement)
- ⚠️ Overfitting (training improves but validation worsens)
- ⚠️ Model instability

## 🛠 **Troubleshooting**

### **Common Issues:**

1. **Training Won't Stop**:
   - Ensure `stop_training.txt` is in correct directory
   - Use `Ctrl+C` as backup method
   - Check GUI connection

2. **Models Not Saving**:
   - Check disk space
   - Verify write permissions to `models/` directory
   - Monitor for error messages

3. **Performance Plateau**:
   - Try different learning rates
   - Adjust network architecture
   - Use different asset combinations

### **Recovery from Crashes:**
- Check for `checkpoint_continuous.pt`
- Resume with: `--mode continuous --model models/checkpoint_continuous.pt`
- Latest regular saves are also available

## 📋 **Best Practices**

### **For 50%+ Return Achievement:**
1. **Use High-Performance Config**: Start with optimized parameters
2. **Monitor Regularly**: Check every 50-100K timesteps
3. **Multiple Runs**: Try different random seeds
4. **Asset Selection**: Use high-volatility stocks (AAPL, NVDA, TSLA)
5. **Patience**: May take several hours of training
6. **Backup Strategy**: Keep multiple checkpoint saves

### **Resource Management:**
- **CPU**: Continuous training uses 1 CPU core intensively
- **Memory**: ~2-4GB RAM depending on network size
- **Disk**: ~50-100MB per saved model
- **Time**: 6-24 hours for high-performance models

## ✅ **Success Metrics**

### **Achievement Indicators:**
- 📈 **Total Return**: 50%+ in backtesting
- 📊 **Sharpe Ratio**: >1.5 for risk-adjusted performance
- 📉 **Max Drawdown**: <20% for risk control
- 🎯 **Win Rate**: >55% for consistency
- ⏱️ **Stability**: Performance maintained across multiple evaluations

**With continuous training, you can now let the model train as long as needed to achieve your 50%+ return target!** 🎯♾️