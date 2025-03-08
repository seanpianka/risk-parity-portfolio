# 📊 Risk Parity Portfolio Optimization

## 📖 Overview: Principles of Risk Parity
This project implements a **Risk Parity portfolio allocation strategy** based on the principles outlined in the AQR Capital paper on Risk Parity. Unlike traditional **60/40 equity-bond portfolios**, which are dominated by equity risk, **Risk Parity** seeks to balance risk contributions **across asset classes**.

### 🔹 **Core Concepts of Risk Parity:**
- **Equalizing Risk Contributions:** Each asset class contributes an equal amount of risk to the portfolio.
- **Volatility-Weighted Allocation:** Riskier assets receive **lower** allocation; lower-volatility assets receive **higher** allocation.
- **Dynamically Adjusted Weights:** Portfolio weights are rebalanced **monthly or quarterly** based on recent risk estimates.
- **Improved Sharpe Ratio:** By maintaining a balanced risk profile, the portfolio seeks to achieve better **risk-adjusted returns**.

### 📉 **How Risk is Measured?**
1. **Rolling Volatility (σ):** Standard deviation of asset returns over the last `N` months.
2. **Rolling Correlation (ρ):** Measures how asset classes move together.
3. **Risk Contribution (RC):** Ensures all assets contribute **equal total risk**.

---

## ⚙️ How This Implementation Works
This executable calculates optimal **Risk Parity weights** for a portfolio of **diverse asset classes** using real-world ticker data.

### **✅ Features:**
- **Rolling Volatility Calculation:** Uses the past `N` months (e.g., 3-6 months) to compute risk.
- **Rolling Correlation Matrix:** Accounts for relationships between assets to avoid over-concentration.
- **Risk Parity Weighting:** Assigns weights to **balance** risk across asset classes.
- **Automated Monthly/Quarterly Rebalancing:** Ensures dynamic adjustments based on recent market conditions.

### **📌 How the Portfolio is Constructed?**
1. **Download Ticker Data** (e.g., Stocks, Bonds, Commodities, REITs, Gold).
2. **Compute Log Returns** for each asset.
3. **Calculate Rolling Volatility & Correlations** (3-6 months lookback).
4. **Construct the Risk Parity Portfolio** using inverse volatility weighting adjusted for correlation.
5. **Normalize Weights & Display Results.**

### **📊 Example Ticker List by Asset Class**
| Asset Class | Example Tickers |
|------------|----------------|
| US Equities | SPY, QQQ       |
| International Equities | EFA, VEU |
| Bonds | TLT, AGG |
| Commodities | GLD, DBC |
| Real Estate | VNQ |
| Crypto | BTC-USD, ETH-USD |

---

## 🏃‍♂️ Running the Risk Parity Optimization
### **🔧 Installation & Setup**
#### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/risk-parity-optimizer.git
cd risk-parity-optimizer
```

#### **2️⃣ Install Dependencies**
Ensure you have Rust installed. If not, install it via [Rustup](https://rustup.rs/):
```bash
cargo install --path .
```

#### **3️⃣ Run the Executable**
```bash
cargo run --release
```

---

## 📈 Example Output
```text
📌 Most Recent Correlation Matrix (Final Rolling Window):
 1.00  -0.71   0.32
-0.71   1.00  -0.58
 0.32  -0.58   1.00

📊 Average Correlation Matrix (Last 5 Periods):
 1.00  -0.65   0.25
-0.65   1.00  -0.52
 0.25  -0.52   1.00

⚖️ Risk Parity Weights: [0.42, 0.33, 0.25]
```

---

## 📆 When to Run This?
This tool is intended to be run **monthly or quarterly** for rebalancing. 

- **Monthly Rebalancing:** Use a shorter lookback window (e.g., 3-6 months).
- **Quarterly Rebalancing:** Use a longer lookback window (e.g., 6-12 months).

By keeping risk allocation **balanced over time**, this approach ensures **optimal performance across market conditions.** 🚀

---

## 🤝 Contributing
Pull requests and suggestions are welcome! Please open an issue if you find any bugs or improvements.

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to modify and distribute.
