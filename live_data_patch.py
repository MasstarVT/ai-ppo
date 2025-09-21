# Patch file to fix live market data
# Get real-time market data
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']

quotes_data = {
    'Symbol': [],
    'Price': [],
    'Change': [],
    'Change %': [],
    'Volume': [],
    'AI Signal': []
}

for symbol in symbols:
    try:
        # Get current price from data client
        current_price = st.session_state.portfolio_manager.get_current_price(symbol)
        
        # Calculate 24h change (simplified - in production would use historical data)
        import random
        change_24h = random.uniform(-5, 5) 
        change_amount = current_price * (change_24h / 100)
        change_pct = f"{change_24h:+.2f}%"
        
        # Mock trading signal and volume
        signal = random.choice(['🟢 BUY', '🔴 SELL', '🟡 HOLD'])
        volume = f"{random.uniform(100, 999):.0f}M"
        
        quotes_data['Symbol'].append(symbol)
        quotes_data['Price'].append(f"${current_price:.2f}")
        quotes_data['Change'].append(f"${change_amount:+.2f}")
        quotes_data['Change %'].append(change_pct)
        quotes_data['Volume'].append(volume)
        quotes_data['AI Signal'].append(signal)
        
    except Exception as e:
        st.error(f"Error fetching real-time data for {symbol}: {e}")
        # Fallback to demo data for this symbol
        quotes_data['Symbol'].append(symbol)
        quotes_data['Price'].append("$0.00")
        quotes_data['Change'].append("$0.00")
        quotes_data['Change %'].append("0.00%")
        quotes_data['Volume'].append("0M")
        quotes_data['AI Signal'].append("🟡 HOLD")

quotes_df = pd.DataFrame(quotes_data)