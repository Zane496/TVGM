import os
import argparse
import pandas as pd
import numpy as np


def calculate_technical_indicators(df, pred_horizon):
    """
    Core Feature Engineering: Calculates 17 technical indicators as defined in the paper.
    Captures intra-day game features, moving averages, momentum, and volume profiles.
    """
    df.columns = [c.lower() for c in df.columns]

    # 1. Base Return (Log Return)
    df['log_ret'] = np.log(df['close'] / (df['close'].shift(1) + 1e-12))

    # 2. Intra-day & High-frequency Game Features
    df['overnight_ret'] = df['open'] / (df['close'].shift(1) + 1e-8) - 1.0
    df['intraday_ret'] = df['close'] / (df['open'] + 1e-8) - 1.0
    df['daily_range'] = (df['high'] - df['low']) / (df['open'] + 1e-8)
    df['close_loc'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['close'] + 1e-8)
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['close'] + 1e-8)

    # 3. Moving Average System (MA5, 10, 20, 60)
    for w in [5, 10, 20, 60]:
        ma = df['close'].rolling(window=w).mean()
        df[f'dist_ma{w}'] = (df['close'] - ma) / (ma + 1e-8)

    # 4. Momentum and Volume Profile
    vol_5 = df['log_ret'].rolling(window=5).std()
    df['adj_mom_5'] = df['log_ret'].rolling(window=5).mean() / (vol_5 + 1e-8)
    vol_ma5 = df['volume'].rolling(window=5).mean()
    df['vol_ratio'] = vol_ma5 / (df['volume'].rolling(window=20).mean() + 1e-8)

    # 5. Institutional Cost Line (VWAP Approx)
    vwap_approx = (df['close'] * df['volume']).rolling(window=5).sum() / (df['volume'].rolling(window=5).sum() + 1e-8)
    df['dist_vwap'] = (df['close'] - vwap_approx) / (vwap_approx + 1e-8)

    # 6. Oscillation and Trend Indicators (RSI, MACD)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / (loss + 1e-8)
    df['rsi_6'] = (100 - (100 / (1 + rs))) / 100.0

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26

    # ======= Target Construction =======
    future_close = df['close'].shift(-pred_horizon)
    df['target'] = (future_close - df['close']) / (df['close'] + 1e-8)
    df['fwd_ret'] = df['target']

    return df


def generate_raw_feature_pool(input_path, output_dir, pred_horizon, top_n):
    """
    Reads raw stock data, aligns dates, calculates features, and saves processing results.
    """
    print(f"[INFO] Loading raw data from {input_path} ...")
    df_all = pd.read_csv(input_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    df_all.columns = [c.lower() for c in df_all.columns]

    # Extract top N valid tickers
    all_tickers = sorted(df_all['name'].unique())[:top_n]
    common_dates = np.array(sorted(df_all['date'].unique()))

    # Feature ordering aligned with paper definitions
    feature_cols = [
        'log_ret', 'overnight_ret', 'intraday_ret', 'daily_range', 'close_loc',
        'upper_shadow', 'lower_shadow', 'dist_ma5', 'dist_ma10', 'dist_ma20',
        'dist_ma60', 'adj_mom_5', 'vol_ratio', 'dist_vwap', 'rsi_6', 'macd',
        'volume',
        'target', 'fwd_ret'  # Target arrays at the end
    ]

    all_features_list = []
    valid_mask_list = []
    final_tickers = []
    gb = df_all.groupby('name')

    print(f"[INFO] Processing features for {len(all_tickers)} stocks...")
    for i, ticker in enumerate(all_tickers):
        if (i + 1) % 50 == 0:
            print(f"       -> Processed {i + 1}/{top_n} stocks")

        df_stock = gb.get_group(ticker).sort_values('date').set_index('date')
        df_aligned = df_stock.reindex(common_dates)

        # Dynamic Masking: Exclude dates with missing current or future close prices
        future_close = df_aligned['close'].shift(-pred_horizon)
        valid_mask = ((~df_aligned['close'].isna().values) &
                      (~future_close.isna().values)).astype(np.float32)

        df_aligned = df_aligned.ffill().reset_index().rename(columns={'index': 'date'})

        # Calculate indicators
        df_ind = calculate_technical_indicators(df_aligned, pred_horizon)
        feats = df_ind[feature_cols].fillna(0.0).values.astype(np.float32)

        all_features_list.append(feats)
        valid_mask_list.append(valid_mask)
        final_tickers.append(ticker)

    # Final shapes: (N, T, F)
    X_data = np.stack(all_features_list, axis=0)
    valid_mask = np.stack(valid_mask_list, axis=0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Feature engineering completed. Shape: {X_data.shape}")

    # Save to processed directory
    np.save(os.path.join(output_dir, 'feature_data.npy'), X_data)
    np.save(os.path.join(output_dir, 'valid_mask.npy'), valid_mask)
    np.save(os.path.join(output_dir, 'stock_names.npy'), np.array(final_tickers))

    print(f"[SUCCESS] Data successfully saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVGM Feature Engineering Pipeline")

    # Directory arguments
    parser.add_argument("--input_path", type=str, default="data/raw/all_stocks_5yr.csv", help="Path to raw csv data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed npy files")

    # Task specific arguments
    parser.add_argument("--task_type", type=str, default="5D", choices=["1D", "5D"], help="Prediction horizon task (1D or 5D)")
    parser.add_argument("--top_n", type=int, default=505, help="Number of stocks to process")

    args = parser.parse_args()

    # Parse horizon
    pred_horizon = 5 if args.task_type == "5D" else 1

    print(f"=== TVGM Feature Processing ===")
    print(f"Task: {args.task_type} | Horizon: {pred_horizon} days | Top N: {args.top_n}")

    generate_raw_feature_pool(
        input_path=args.input_path,
        output_dir=args.output_dir,
        pred_horizon=pred_horizon,
        top_n=args.top_n
    )