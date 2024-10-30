import pandas as pd
import numpy as np
from datetime import datetime
import os
import requests
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from io import StringIO
import talib
import yfinance as yf
from tqdm import tqdm

from .config import TWStockConfig, MarketDateRange

class DataPreprocessor:
    """數據預處理工具類"""
    def __init__(self, config: TWStockConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """設定日誌系統"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 移除現有的處理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 檔案處理器
        file_handler = logging.FileHandler(
            self.config.meta_data_dir / 'market_data_process.log',
            encoding='utf-8'
        )
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        # 控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def preprocess_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """預處理股票數據，處理缺失值和類型轉換"""
        df = df.copy()
        df['證券代號'] = df['證券代號'].astype(str)
        
        numeric_columns = [
            '成交股數', '成交筆數', '成交金額', '開盤價', 
            '最高價', '最低價', '收盤價', '漲跌價差',
            '最後揭示買價', '最後揭示買量', 
            '最後揭示賣價', '最後揭示賣量', '本益比'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].replace('--', np.nan)
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


class TechnicalAnalyzer:
    """技術分析工具類"""
    def __init__(self, config: TWStockConfig):
        self.config = config
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算技術指標"""
        df = df.copy()
        
        if len(df) < self.config.min_data_days:
            return df
            
        # 確保價格數據為數值型
        price_cols = ['收盤價', '最高價', '最低價']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 移除所有價格都是 NaN 的列
        df = df.dropna(subset=price_cols, how='all')
        
        # 基礎技術指標
        df['SMA30'] = talib.SMA(df['收盤價'], timeperiod=30)
        df['DEMA30'] = talib.DEMA(df['收盤價'], timeperiod=30)
        df['EMA30'] = talib.EMA(df['收盤價'], timeperiod=30)
        df['RSI'] = talib.RSI(df['收盤價'], timeperiod=14)
        
        # MACD
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['收盤價'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # 需要高低價的指標
        if not df[['最高價', '最低價', '收盤價']].isnull().any().any():
            df['slowk'], df['slowd'] = talib.STOCH(
                df['最高價'], df['最低價'], df['收盤價'],
                fastk_period=5, slowk_period=3, slowk_matype=0,
                slowd_period=3, slowd_matype=0
            )
            df['SAR'] = talib.SAR(df['最高價'], df['最低價'], acceleration=0.02, maximum=0.2)
        
        # 趨勢指標
        df['TSF'] = talib.TSF(df['收盤價'], timeperiod=14)
        df['middleband'], _, _ = talib.BBANDS(
            df['收盤價'], timeperiod=30, nbdevup=2, nbdevdn=2, matype=0
        )
        
        return df


class MarketDataProcessor:
    """市場數據處理器"""
    def __init__(self, config: TWStockConfig, date_range: Optional[MarketDateRange] = None):
        self.config = config
        self.date_range = date_range or MarketDateRange()
        self.preprocessor = DataPreprocessor(config)
        self.analyzer = TechnicalAnalyzer(config)
        
        # 記錄設定的日期範圍
        self.preprocessor.logger.info(f"設定數據處理範圍: {self.date_range.date_range_str}")
    
    def cleanup_old_backups(self, keep_days: int = 7):
        """清理舊的備份檔案"""
        try:
            backup_dir = self.config.backup_dir
            if not backup_dir.exists():
                return
            
            # 取得所有備份檔案
            backup_files = list(backup_dir.glob('*.csv'))
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            
            # 刪除超過保留天數的檔案
            if len(backup_files) > keep_days:
                for old_file in backup_files[:-keep_days]:
                    try:
                        old_file.unlink()
                        self.preprocessor.logger.info(f"Removed old backup: {old_file}")
                    except Exception as e:
                        self.preprocessor.logger.error(f"Failed to remove {old_file}: {e}")
        
        except Exception as e:
            self.preprocessor.logger.error(f"Error during backup cleanup: {e}")

    def process_daily_stock_data(self) -> bool:
        """處理每日股票數據並計算技術指標"""
        try:
            # 讀取原始資料
            self.preprocessor.logger.info("Reading stock data...")
            stock_data = pd.read_csv(self.config.stock_data_file, low_memory=False)
            
            # 準備備份
            backup_dir = self.config.backup_dir
            backup_dir.mkdir(exist_ok=True)
            today = datetime.now().strftime('%Y%m%d')
            backup_path = backup_dir / f'stock_data_{today}.csv'
            
            if not backup_path.exists():
                shutil.copy2(self.config.stock_data_file, backup_path)
                self.cleanup_old_backups(self.config.backup_keep_days)
            
            # 按股票代號分組處理
            processed_data = []
            total_stocks = len(stock_data['證券代號'].unique())
            
            for idx, (stock_id, group) in enumerate(stock_data.groupby('證券代號'), 1):
                try:
                    # 預處理數據
                    df = self.preprocessor.preprocess_stock_data(group)
                    
                    # 計算技術指標
                    if len(df) >= self.config.min_data_days:
                        df = self.analyzer.calculate_indicators(df)
                        processed_data.append(df)
                        
                        # 儲存個別股票的指標檔案
                        output_path = self.config.get_technical_file(stock_id)
                        df.to_csv(output_path, index=False, encoding='utf-8-sig')
                    else:
                        self.preprocessor.logger.warning(f"Insufficient data for stock ID {stock_id}")
                    
                    self.preprocessor.logger.info(f"Processed {idx}/{total_stocks} - Stock ID: {stock_id}")
                    
                except Exception as e:
                    self.preprocessor.logger.error(f"Error processing stock {stock_id}: {str(e)}")
                    continue
            
            if processed_data:
                # 合併所有處理後的資料
                self.preprocessor.logger.info("Merging processed data...")
                all_stocks_df = pd.concat(processed_data, ignore_index=True)
                
                # 儲存完整的處理後資料
                output_path = self.config.meta_data_dir / 'all_stocks_indicators.csv'
                all_stocks_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                self.preprocessor.logger.info("Processing completed successfully")
                return True
            
            self.preprocessor.logger.warning("No data was processed")
            return False
            
        except Exception as e:
            self.preprocessor.logger.error(f"Error during processing: {str(e)}")
            return False