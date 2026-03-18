import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed
import logging
import warnings
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=Warning, message=".*An input array is constant.*")

def compute_lag_correlations(target_series, cov_series, cov_name, lags):
    """计算单个特征的滞后相关性"""
    results = []
    target_rank = target_series.rank()
    for lag in range(1, lags + 1):
        shifted_cov = cov_series.shift(lag)
        if shifted_cov.nunique(dropna=True) <= 1 or target_series.nunique(dropna=True) <= 1:
            results.append((cov_name, lag, np.nan))
            continue
        corr, _ = spearmanr(target_rank, shifted_cov.rank(), nan_policy='omit')
        results.append((cov_name, lag, corr))
    return results

def lag_correlation_analysis(df, target_col, lags=24, n_jobs=12):
    """通用的多线程滞后相关性分析流水线"""
    logging.info(f"Starting lag correlation analysis for target: {target_col}")
    target_series = df[target_col]
    covariate_cols = [col for col in df.columns if col.startswith('信号') and col != target_col]
    
    # 使用 Joblib 并行计算
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_lag_correlations)(target_series, df[col], col, lags) 
        for col in covariate_cols
    )
    
    # 展平结果并返回分析报告
    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flat_results, columns=['Covariate', 'Lag', 'Spearman_Corr'])
    results_df = results_df.dropna().sort_values(by='Spearman_Corr', ascending=False)
    
    logging.info("Analysis complete.")
    return results_df

if __name__ == "__main__":

    DATA_PATH = "data/sample_data.csv"
    TARGET_COL = "信号123" # 这里可以灵活改为 "信号124" 等任何目标列
    
    if os.path.exists(DATA_PATH):
        df = pd.read_excel(DATA_PATH)
        # 执行特征工程分析
        corr_report = lag_correlation_analysis(df, target_col=TARGET_COL, lags=24)
        print(f"Top 5 correlated features for {TARGET_COL}:")
        print(corr_report.head())
    else:
        print(f"Data not found at {DATA_PATH}. Please provide valid data.")