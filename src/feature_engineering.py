import os
import logging
import warnings
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from joblib import Parallel, delayed

# 忽略常数输入数组产生的警告 (工业传感器在特定时间段内数值无变化属于正常现象)
warnings.filterwarnings("ignore", category=Warning, message=".*An input array is constant.*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def compute_lag_correlations(target_series, cov_series, cov_name, max_lag):
    """
    计算单个传感器协变量在不同滞后阶数下与目标变量的 Spearman 秩相关系数。
    
    参数:
        target_series: 目标变量的时间序列
        cov_series: 待评估协变量的时间序列
        cov_name: 协变量的特征名称
        max_lag: 最大滞后阶数
    """
    results = []
    # 提前计算目标变量的 rank，避免在滞后循环中重复计算，降低时间开销
    target_rank = target_series.rank()
    
    for lag in range(1, max_lag + 1):
        shifted_cov = cov_series.shift(lag)
        
        # 过滤掉平滑或失效的传感器序列（即在当前窗口内方差为0，无有效信息）
        if shifted_cov.nunique(dropna=True) <= 1 or target_series.nunique(dropna=True) <= 1:
            results.append((cov_name, lag, np.nan))
            continue
            
        # 计算 Spearman 相关性，忽略因 shift 产生的 NaN 值
        corr, _ = spearmanr(target_rank, shifted_cov.rank(), nan_policy='omit')
        results.append((cov_name, lag, corr))
        
    return results

def lag_correlation_analysis(df, target_col, max_lag=24, n_jobs=12):
    """
    多进程计算所有传感器信号的滞后相关性，用于后续的 CNN-LSTM 模型特征筛选。
    """
    logging.info(f"开始计算目标变量 {target_col} 的特征滞后相关性 (max_lag={max_lag})")
    
    target_series = df[target_col]
    # 筛选出所有传感器信号列，并排除目标变量本身避免数据穿越
    covariate_cols = [col for col in df.columns if col.startswith('信号') and col != target_col]
    
    # 使用 joblib 并行加速特征遍历
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_lag_correlations)(target_series, df[col], col, max_lag) 
        for col in covariate_cols
    )
    
    # 展平嵌套的并行结果列表
    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flat_results, columns=['Covariate', 'Lag', 'Spearman_Corr'])
    
    # 剔除无效值，并按照相关系数的绝对值进行降序排列（强负相关同样具备特征价值）
    results_df = results_df.dropna().sort_values(by='Spearman_Corr', ascending=False, key=abs)
    
    logging.info(f"特征相关性分析完成，共评估 {len(covariate_cols)} 个有效信号。")
    return results_df

if __name__ == "__main__":
    
    data_path = "data/sample_data.csv"
    target_signal = "信号123" 
    
    if os.path.exists(data_path):
        logging.info(f"加载数据集: {data_path}")
        # 注意：此处根据实际数据后缀选择 read_csv 或 read_excel
        df = pd.read_excel(data_path) 
        
        corr_report = lag_correlation_analysis(df, target_col=target_signal, max_lag=24)
        
        print(f"\n[{target_signal}] Top 5 高度相关特征 (按绝对值):")
        print(corr_report.head())
    else:
        logging.error(f"找不到数据文件: {data_path}，请检查挂载路径。")
