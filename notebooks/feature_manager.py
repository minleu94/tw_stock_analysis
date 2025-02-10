import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import warnings
from tqdm import tqdm
import gc
import psutil
import talib

from feature_config import FeatureConfig

class FeatureManager:
    """特徵管理系統"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 初始化日誌
        self._initialize_logging()
        
        # 驗證配置
        if not self.validate_config():
            raise ValueError("配置驗證失敗")
            
    def _initialize_logging(self):
        """初始化日誌系統"""
        if not self.logger.handlers:
            # 檔案處理器
            log_file = self.config.log_path / f'feature_generator_{datetime.now():%Y%m%d}.log'
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            # 控制台處理器
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)
    
    def validate_config(self) -> bool:
        """驗證配置"""
        try:
            # 驗證路徑
            required_paths = [
                self.config.base_path,
                self.config.meta_data_path,
                self.config.test_data_path,
                self.config.features_path
            ]
            
            for path in required_paths:
                if not path.exists():
                    self.logger.error(f"找不到必要目錄: {path}")
                    return False
            
            # 驗證必要文件
            required_files = [
                self.config.meta_data_path / 'companies.csv',
                self.config.meta_data_path / 'industry_index.csv',
                self.config.test_data_path / 'test_stock_data.csv'
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    self.logger.error(f"找不到必要文件: {file_path}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"驗證配置時發生錯誤: {str(e)}")
            return False
    
    def generate_features(self) -> bool:
        """生成特徵"""
        try:
            # 備份重要文件
            self.config.backup_important_files()
            
            # 讀取股票主檔
            main_df = pd.read_csv(
                self.config.get_stock_data_path(),
                dtype={'證券代號': str, '證券名稱': str}
            )
            
            # 處理每支股票
            for stock_id in self.config.TEST_SETTING['test_stocks']:
                try:
                    # 處理單一股票
                    stock_df = main_df[main_df['證券代號'] == stock_id].copy()
                    if stock_df.empty:
                        self.logger.warning(f"找不到股票 {stock_id} 的資料")
                        continue
                        
                    # 生成特徵
                    result_df = self._generate_stock_features(stock_df, stock_id)
                    if result_df is not None:
                        # 儲存特徵
                        self._save_features(result_df, stock_id)
                        
                except Exception as e:
                    self.logger.error(f"處理股票 {stock_id} 時發生錯誤: {str(e)}")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"生成特徵時發生錯誤: {str(e)}")
            return False
    
    def _generate_stock_features(self, df: pd.DataFrame, stock_id: str) -> Optional[pd.DataFrame]:
        """生成單一股票的特徵"""
        try:
            # 數據預處理
            df = self._preprocess_data(df)
            if df is None:
                return None
            
            # 添加技術特徵
            df = self._add_technical_features(df)
            df = self._add_volume_features(df)
            df = self._add_trend_features(df)
            
            # 後處理
            df = self._post_process_features(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"生成股票 {stock_id} 特徵時發生錯誤: {str(e)}")
            return None
    
    def _preprocess_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """數據預處理"""
        try:
            df = df.copy()
            
            # 處理日期格式
            df['日期'] = pd.to_datetime(df['日期'])
            
            # 過濾日期範圍
            mask = (df['日期'] >= pd.to_datetime(self.config.TEST_SETTING['start_date'])) & \
                   (df['日期'] <= pd.to_datetime(self.config.TEST_SETTING['end_date']))
            df = df[mask]
            
            # 處理數值欄位
            numeric_columns = ['開盤價', '最高價', '最低價', '收盤價', '成交股數', '成交金額']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            self.logger.error(f"數據預處理時發生錯誤: {str(e)}")
            return None
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技術特徵"""
        try:
            params = self.config.FEATURE_PARAMS['technical']
            
            # 計算RSI
            df['RSI'] = talib.RSI(df['收盤價'], timeperiod=params['rsi_period'])
            
            # 計算MACD
            macd_params = params['macd_params']
            df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
                df['收盤價'],
                fastperiod=macd_params['fast_period'],
                slowperiod=macd_params['slow_period'],
                signalperiod=macd_params['signal_period']
            )
            
            # 計算KD
            kd_params = params['kd_params']
            df['slowk'], df['slowd'] = talib.STOCH(
                df['最高價'],
                df['最低價'],
                df['收盤價'],
                fastk_period=kd_params['fastk_period'],
                slowk_period=kd_params['slowk_period'],
                slowk_matype=0,
                slowd_period=kd_params['slowd_period'],
                slowd_matype=0
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加技術特徵時發生錯誤: {str(e)}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特徵"""
        try:
            params = self.config.FEATURE_PARAMS['volume']
            
            # 計算量比
            df['量比'] = df['成交股數'] / df['成交股數'].rolling(params['long_period']).mean()
            
            # 計算量增率
            df['量增率'] = df['成交股數'].pct_change()
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加成交量特徵時發生錯誤: {str(e)}")
            return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加趨勢特徵"""
        try:
            params = self.config.FEATURE_PARAMS['trend']
            
            # 計算移動平均
            df['MA20'] = talib.SMA(df['收盤價'], timeperiod=params['ma_period'])
            
            # 計算趨勢強度
            df['趨勢強度'] = (df['收盤價'] - df['MA20']) / df['MA20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加趨勢特徵時發生錯誤: {str(e)}")
            return df
    
    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徵後處理"""
        try:
            # 處理無限值
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 填補空值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
            
            return df
            
        except Exception as e:
            self.logger.error(f"特徵後處理時發生錯誤: {str(e)}")
            return df
    
    def _save_features(self, df: pd.DataFrame, stock_id: str) -> bool:
        """儲存特徵"""
        try:
            # 生成文件名
            start_date = df['日期'].min().strftime('%Y%m%d')
            end_date = df['日期'].max().strftime('%Y%m%d')
            filename = f"{stock_id}_{start_date}_{end_date}.csv"
            
            # 儲存文件
            save_path = self.config.features_path / filename
            df.to_csv(save_path, index=False, encoding='utf-8')
            
            self.logger.info(f"已儲存特徵至: {save_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"儲存特徵時發生錯誤: {str(e)}")
            return False
    
    def _check_memory_usage(self):
        """檢查記憶體使用情況"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_gb = memory_info.rss / (1024 * 1024 * 1024)
            
            if memory_usage_gb > 8:  # 8GB
                self.logger.warning(f"記憶體使用量過高: {memory_usage_gb:.2f}GB")
                gc.collect()
                
        except Exception as e:
            self.logger.error(f"檢查記憶體使用時發生錯誤: {str(e)}") 