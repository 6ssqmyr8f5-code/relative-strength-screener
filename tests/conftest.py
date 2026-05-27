import numpy as np
import pandas as pd


def make_synthetic_df(kind: str) -> pd.DataFrame:
    n = 320
    dates = pd.date_range(start="2022-01-01", periods=n, freq="D")

    base_stock = 100.0
    base_index = 100.0
    stock_prices = [base_stock]
    index_prices = [base_index]
    volumes = [1000000] * n

    for i in range(1, n):
        if kind == "a_type":
            drift = -0.0003 * i + 0.001 * (i % 20)
            noise = np.random.randn() * 0.015
            stock_change = drift + noise
            index_change = -0.0001 * i + np.random.randn() * 0.012
        elif kind == "a_plus":
            drift = -0.0002 * i + 0.0015 * (i % 15)
            noise = np.random.randn() * 0.012
            stock_change = drift + noise
            index_change = -0.00008 * i + np.random.randn() * 0.01
        elif kind == "b_type":
            drift = 0.0002 * i
            noise = np.random.randn() * 0.013
            stock_change = drift + noise
            index_change = 0.0001 * i + np.random.randn() * 0.011
        elif kind == "c_type":
            drift = 0.0005 * i
            noise = np.random.randn() * 0.018
            stock_change = drift + noise
            index_change = 0.0003 * i + np.random.randn() * 0.014
        elif kind == "c_hot":
            drift = 0.0008 * i
            noise = np.random.randn() * 0.02
            stock_change = drift + noise
            index_change = 0.00015 * i + np.random.randn() * 0.012
        elif kind == "reject":
            drift = -0.0001 * i + np.random.randn() * 0.02
            stock_change = drift
            index_change = 0.0002 * i + np.random.randn() * 0.015
        elif kind == "insufficient":
            n_short = 100
            dates = pd.date_range(start="2024-01-01", periods=n_short, freq="D")
            stock_prices = [100 + i * 0.1 + np.random.randn() * 0.5 for i in range(n_short)]
            index_prices = [100 + i * 0.05 + np.random.randn() * 0.3 for i in range(n_short)]
            volumes = [1000000] * n_short
            df = pd.DataFrame({
                "date": dates,
                "stock_close": stock_prices,
                "index_close": index_prices,
                "volume": volumes,
            })
            return df
        else:
            raise ValueError(f"Unknown kind: {kind}")

        new_stock = stock_prices[-1] * (1 + stock_change)
        new_index = index_prices[-1] * (1 + index_change)
        stock_prices.append(new_stock)
        index_prices.append(new_index)

    df = pd.DataFrame({
        "date": dates,
        "stock_close": stock_prices,
        "index_close": index_prices,
        "volume": volumes,
    })
    return df