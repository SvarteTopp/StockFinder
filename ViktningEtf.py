import yfinance as yf
import pandas as pd
import numpy as np

def calculate_portfolio(weights, years_back, rebalance_freq):
    """
    Beräkna portföljens utveckling baserat på historisk data.
    
    :param weights: Dictionary med vikter för varje ETF (t.ex. {'QQQ': 0.4, 'VOO': 0.4, 'SCHD': 0.2})
    :param years_back: Antal år bakåt att beräkna (t.ex. 5)
    :param rebalance_freq: Hur ofta ombalansering sker per år (0–12)
    :return: DataFrame med portföljens värde över tid
    """
    # Hämta historisk data
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(years=years_back)
    
    tickers = list(weights.keys())
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # Beräkna daglig avkastning
    returns = data.pct_change().dropna()
    
    # Skapa en portfölj
    portfolio_value = 10000  # Startvärde på portföljen
    portfolio_values = [portfolio_value]
    current_weights = np.array(list(weights.values()))
    
    for i in range(1, len(returns)):
        # Uppdatera portföljens värde
        daily_return = np.sum(returns.iloc[i] * current_weights)
        portfolio_value *= (1 + daily_return)
        portfolio_values.append(portfolio_value)
        
        # Ombalansera vid angiven frekvens
        if rebalance_freq > 0 and i % (252 // rebalance_freq) == 0:
            current_weights = np.array(list(weights.values()))
    
    # Skapa en DataFrame med resultat
    result = pd.DataFrame({'Datum': returns.index, 'Portföljvärde': portfolio_values})
    return result

# Exempel på användning
weights = {'QQQ': 0.4, 'VOO': 0.4, 'SCHD': 0.2}  # 40% QQQ, 40% VOO, 20% SCHD
years_back = 5  # Beräkna för de senaste 5 åren
rebalance_freq = 2  # Ombalansera 2 gånger per år

result = calculate_portfolio(weights, years_back, rebalance_freq)
print(result)

# Visualisera resultatet
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(result['Datum'], result['Portföljvärde'], label='Portföljvärde')
plt.title('Portföljens utveckling över tid')
plt.xlabel('Datum')
plt.ylabel('Värde (SEK)')
plt.legend()
plt.grid()
plt.show()
