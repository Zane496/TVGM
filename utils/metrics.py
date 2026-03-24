import numpy as np


def calculate_metrics(period_returns, precision_list, market_precision_list, horizon=5, mkt_baseline=0.5474):
    """
    Calculate the 7 core metrics used in the paper for financial evaluation.

    Args:
        period_returns (list): List of portfolio returns for each rebalancing period.
        precision_list (list): List of Hit Rates (P@K) for each period.
        market_precision_list (list): List of market average hit rates (optional if using fixed baseline).
        ic_list (list): List of cross-sectional Information Coefficients.
        horizon (int): The prediction horizon (e.g., 5 for weekly rebalancing).
        mkt_baseline (float): The fixed global market win rate baseline for LiftPct calculation.

    Returns:
        list: [Sharpe, CumRet, AnnRet, AnnVol, MaxDD, P@K, LiftPct]
    """
    R = np.array(period_returns)
    if len(R) == 0:
        return [0.0] * 8

    # 1. AnnRet (Annualized Return)
    ann_ret = np.mean(R) * (252 / horizon)

    # 2. AnnVol (Annualized Volatility)
    ann_vol = np.std(R) * np.sqrt(252 / horizon)

    # 3. Sharpe Ratio
    sharpe = ann_ret / (ann_vol + 1e-8)

    # 4. CumRet (Cumulative Return)
    nav = np.cumprod(1 + R)
    cum_ret = nav[-1] - 1

    # 5. MaxDD (Maximum Drawdown)
    dd = (nav - np.maximum.accumulate(nav)) / (np.maximum.accumulate(nav) + 1e-8)
    max_dd = dd.min()

    # 6. P@K (Hit Rate)
    p_at_k = np.nanmean(precision_list) if precision_list else 0.0

    # 7. LiftPct (Lift Percentage over market baseline)
    lift_pct = (p_at_k - mkt_baseline) / (mkt_baseline + 1e-8)


    return [sharpe, cum_ret, ann_ret, ann_vol, max_dd, p_at_k, lift_pct]
