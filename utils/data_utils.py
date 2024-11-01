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
    def __init__(self, base_path="D:\\Min\\Python\\Project\\FA_Data"):
        self.base_path = base_path
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """設置日誌"""
        logger = logging.getLogger('DataPreprocessor')
        logger.setLevel(logging.INFO)
        
        # 確保處理器不會重複添加
        if not logger.handlers:
            # 檔案處理器
            fh = logging.FileHandler('data_processing.log', encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # 控制台處理器
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # 格式化
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            # 添加處理器
            logger.addHandler(fh)
            logger.addHandler(ch)
        
        return logger

    @staticmethod
    def process_price_data(df):
        """處理價格資料，包含清理千分位逗號和特殊字符"""
        df = df.copy()
        price_columns = ['開盤價', '最高價', '最低價', '收盤價', '最後揭示買價', '最後揭示賣價']
        
        for col in price_columns:
            if col in df.columns:
                # 1. 移除千分位逗號
                df[col] = df[col].astype(str).str.replace(',', '')
                
                # 2. 處理特殊字符
                df[col] = df[col].replace('--', np.nan)
                df[col] = df[col].replace('', np.nan)
                
                # 3. 轉換為浮點數
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def process_volume_data(self, df):
        """處理成交量相關數據"""
        df = df.copy()
        volume_columns = ['成交股數', '成交筆數', '成交金額', '最後揭示買量', '最後揭示賣量']
        
        for col in volume_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df

    def process_date_data(self, df):
        """處理日期數據"""
        if '日期' in df.columns:
            # 確保日期格式統一
            df['日期'] = pd.to_datetime(df['日期'])
        return df

    def clean_stock_id(self, df):
        """清理股票代碼格式"""
        if '證券代號' in df.columns:
            df['證券代號'] = df['證券代號'].astype(str).str.strip()
        return df

    def validate_data(self, df):
        """驗證數據的完整性"""
        required_columns = ['證券代號', '證券名稱', '日期', '收盤價']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # 檢查是否有足夠的數據列
        if len(df) == 0:
            self.logger.error("Empty dataframe")
            return False
            
        return True

    def preprocess_data(self, df):
        """完整的數據預處理流程"""
        try:
            # 驗證數據
            if not self.validate_data(df):
                return None

            # 複製數據避免修改原始數據
            df = df.copy()
            
            # 依序處理各種數據
            df = self.clean_stock_id(df)
            df = self.process_date_data(df)
            df = self.process_price_data(df)
            df = self.process_volume_data(df)
            
            # 排序數據
            if '日期' in df.columns and '證券代號' in df.columns:
                df = df.sort_values(['證券代號', '日期'])
            
            # 移除完全重複的行
            df = df.drop_duplicates()
            
            self.logger.info(f"Successfully preprocessed data: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing data: {str(e)}")
            return None

    def save_data(self, df, filename, subfolder=''):
        """儲存處理後的數據"""
        try:
            # 建立完整的儲存路徑
            save_path = os.path.join(self.base_path, subfolder)
            os.makedirs(save_path, exist_ok=True)
            
            full_path = os.path.join(save_path, filename)
            
            # 儲存數據
            df.to_csv(full_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Data saved successfully to {full_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False

    def backup_data(self, df, filename):
        """備份數據"""
        try:
            # 創建備份目錄
            backup_dir = os.path.join(self.base_path, 'backup')
            os.makedirs(backup_dir, exist_ok=True)
            
            # 生成備份檔案名稱（加入時間戳）
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{os.path.splitext(filename)[0]}_{timestamp}.csv"
            
            # 儲存備份
            backup_path = os.path.join(backup_dir, backup_filename)
            df.to_csv(backup_path, index=False, encoding='utf-8-sig')
            
            self.logger.info(f"Backup created successfully: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False

    def clean_old_backups(self, days_to_keep=7):
        """清理舊的備份檔案"""
        try:
            backup_dir = os.path.join(self.base_path, 'backup')
            if not os.path.exists(backup_dir):
                return True
                
            # 取得所有備份檔案
            backup_files = [f for f in os.listdir(backup_dir) if f.endswith('.csv')]
            backup_files.sort()
            
            # 如果備份檔案數量超過指定天數，刪除最舊的檔案
            while len(backup_files) > days_to_keep:
                oldest_file = os.path.join(backup_dir, backup_files[0])
                os.remove(oldest_file)
                self.logger.info(f"Removed old backup: {oldest_file}")
                backup_files = backup_files[1:]
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error cleaning old backups: {str(e)}")
            return False


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