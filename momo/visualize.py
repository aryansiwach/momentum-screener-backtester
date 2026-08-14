
import matplotlib.pyplot as plt
import pandas as pd

class Visualizer:
    """Handles all charting for momentum pipeline."""
    
    @staticmethod
    def equity_curve(equity: pd.Series, benchmark: pd.Series = None):
        plt.figure(figsize=(10,4))
        if benchmark is not None:
            pd.DataFrame({"Strategy": equity, "Benchmark": benchmark}).plot(ax=plt.gca())
        else:
            equity.plot(ax=plt.gca())
        plt.title("Equity Curve")
        plt.xlabel("Date"); plt.ylabel("Growth of $1")
        plt.tight_layout(); plt.show()

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window=63, ppy=252):
        rs = (returns.rolling(window).mean() * ppy) / (returns.rolling(window).std() * (ppy**0.5))
        plt.figure(figsize=(10,3))
        rs.plot()
        plt.title(f"Rolling Sharpe (window={window})")
        plt.tight_layout(); plt.show()

    @staticmethod
    def score_heatmap(scores: pd.DataFrame, n_last=100):
        plt.figure(figsize=(10,4))
        subset = scores.tail(n_last)
        plt.imshow(subset.T, aspect='auto', cmap='coolwarm', interpolation='none')
        plt.colorbar(label='Momentum Score')
        plt.title("Composite Momentum Scores (recent)")
        plt.yticks(range(len(subset.columns)), subset.columns)
        plt.tight_layout(); plt.show()
