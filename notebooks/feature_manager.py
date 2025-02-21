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
import talib as ta
import json
import os

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
            
        # 讀取產業對照檔案
        self.industry_mapping = pd.read_csv(
            self.config.meta_data_path / 'industry_mapping_analysis.csv',
            encoding=self.config.ENCODING
        )
            
        self.column_mapping = {
            '成交股數': 'volume',
            '開盤價': 'open',
            '最高價': 'high',
            '最低價': 'low',
            '收盤價': 'close',
            '成交金額': 'amount',
            '漲跌價差': 'change',
            '成交筆數': 'transactions',
            '證券代號': 'stock_id',
            '證券名稱': 'stock_name'
        }
        
        # 讀取產業指數數據
        self.industry_index = pd.read_csv(
            self.config.meta_data_path / 'industry_index.csv',
            encoding=self.config.ENCODING
        )
        
    def _initialize_logging(self):
        """初始化日誌系統"""
        if not self.logger.handlers:
            # 檔案處理器
            log_file = self.config.log_path / f'feature_manager_{datetime.now():%Y%m%d}.log'
            fh = logging.FileHandler(log_file, encoding=self.config.ENCODING)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            
            # 控制台處理器
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('INFO: %(message)s'))
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
            self.logger.setLevel(logging.INFO)
    
    def validate_config(self) -> bool:
        """驗證配置"""
        try:
            # 驗證路徑
            required_paths = [
                self.config.project_path,
                self.config.data_path,
                self.config.meta_data_path,
                self.config.test_data_path,
                self.config.features_path,
                self.config.technical_analysis_path,
                self.config.industry_analysis_path,
                self.config.industry_correlation_path,
                self.config.log_path
            ]
            
            # 確保所有必要的路徑都存在
            for path in required_paths:
                if not isinstance(path, Path):
                    self.logger.error(f"路徑對象類型錯誤: {path}")
                    return False
                if not path.exists():
                    try:
                        path.mkdir(parents=True, exist_ok=True)
                        self.logger.info(f"創建目錄: {path}")
                    except Exception as e:
                        self.logger.error(f"創建目錄失敗: {path}, 錯誤: {str(e)}")
                        return False
            
            # 驗證必要文件
            required_files = [
                self.config.meta_data_path / 'companies.csv',
                self.config.meta_data_path / 'industry_index.csv',
                self.config.meta_data_path / 'industry_mapping_analysis.csv',
                self.config.meta_data_path / 'market_index.csv',
                self.config.get_stock_data_path()
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    self.logger.error(f"找不到必要文件: {file_path}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"驗證配置時發生錯誤: {str(e)}")
            return False
    
    def _get_industry_file_name(self, industry_name: str) -> str:
        """
        從產業對照檔案中取得對應的分析檔案名稱
        Args:
            industry_name: 產業名稱
        Returns:
            str: 產業分析檔案名稱
        """
        mapping_df = pd.read_csv(self.config.meta_data_path / 'industry_mapping_analysis.csv')
        matched_row = mapping_df[mapping_df['標準化產業'] == industry_name]
        
        if not matched_row.empty:
            return matched_row['產業分析檔案'].iloc[0]
        else:
            raise ValueError(f"找不到產業 {industry_name} 的對應分析檔案名稱")

    def generate_features(self) -> bool:
        """生成特徵"""
        try:
            self.logger.info("開始生成特徵...")
            
            # 遍歷產業對照檔案
            for _, row in self.industry_mapping.iterrows():
                industry_name = row['標準化產業']
                stock_id = row['對應產業指數']
                
                if pd.isna(stock_id) or stock_id == '未找到對應指數':
                    self.logger.info(f"{industry_name}: 無對應指數，跳過處理")
                    continue
                    
                try:
                    # 讀取產業分析資料
                    start_date = self.config.TEST_SETTING['start_date'].replace('-', '')
                    end_date = self.config.TEST_SETTING['end_date'].replace('-', '')
                    
                    # 構建檔案路徑
                    analysis_path = self.config.get_industry_price_index_path(
                        industry_name=industry_name
                    )
                    
                    if not analysis_path.exists():
                        self.logger.info(f"找不到產業分析檔案: {analysis_path}")
                        continue
                        
                    # 嘗試不同的編碼方式讀取檔案
                    encodings = ['utf-8-sig', 'utf-8', 'big5', 'cp950']
                    industry_data = None
                    
                    for encoding in encodings:
                        try:
                            with open(analysis_path, 'r', encoding=encoding) as f:
                                industry_data = json.load(f)
                                # 檢查是否成功讀取到正確的數據結構
                                if ('time_series_analysis' in industry_data and 
                                    'trend' in industry_data['time_series_analysis'] and
                                    'price_range' in industry_data['time_series_analysis']['trend']):
                                    break
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                            
                    if industry_data is None:
                        self.logger.error(f"無法讀取產業分析檔案: {analysis_path}")
                        continue
                        
                    # 檢查數據結構
                    if 'time_series_analysis' not in industry_data:
                        self.logger.error(f"產業分析檔案缺少time_series_analysis: {analysis_path}")
                        continue
                        
                    # 從time_series_analysis中提取價格數據
                    trend_data = industry_data['time_series_analysis']['trend']
                    if 'price_range' not in trend_data:
                        self.logger.error(f"產業分析檔案缺少price_range: {analysis_path}")
                        continue
                        
                    # 創建時間序列數據
                    start_date = pd.to_datetime(self.config.TEST_SETTING['start_date'])
                    end_date = pd.to_datetime(self.config.TEST_SETTING['end_date'])
                    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
                    
                    # 使用線性插值創建價格序列
                    start_price = float(trend_data['price_range']['start_price'])
                    end_price = float(trend_data['price_range']['end_price'])
                    slope = float(trend_data['slope'])
                    
                    # 創建價格序列
                    days = len(date_range)
                    price_series = np.linspace(start_price, end_price, days)
                    
                    # 創建DataFrame
                    df = pd.DataFrame({
                        '日期': date_range,
                        'close': price_series
                    })
                    df.set_index('日期', inplace=True)
                    
                    # 生成其他價格數據
                    df['open'] = df['close'].shift(1).bfill()
                    df['high'] = df['close'] * 1.01  # 假設每日最高價比收盤價高1%
                    df['low'] = df['close'] * 0.99   # 假設每日最低價比收盤價低1%
                    df['volume'] = 1000000  # 使用固定成交量
                    
                    # 重置索引，確保日期欄位存在
                    df.reset_index(inplace=True)
                    
                    # 計算特徵
                    features = {}
                    for i, row in df.iterrows():
                        date_str = row['日期'].strftime('%Y-%m-%d')
                        
                        # 基本價格特徵
                        features[date_str] = {
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': float(row['volume'])
                        }
                        
                        # 計算技術指標
                        if i >= 30:  # 確保有足夠的歷史數據
                            tech_features = self._calculate_technical_indicators(
                                df['close'].values[:i+1],
                                df['high'].values[:i+1],
                                df['low'].values[:i+1],
                                df['volume'].values[:i+1]
                            )
                            features[date_str].update(tech_features)
                    
                    # 轉換特徵為DataFrame格式
                    feature_df = pd.DataFrame.from_dict(features, orient='index')
                    feature_df.index.name = '日期'
                    feature_df.reset_index(inplace=True)
                    feature_df['stock_id'] = stock_id
                    feature_df['stock_name'] = industry_name
                    
                    # 儲存特徵
                    feature_path = self.config.features_path / self.config.get_feature_filename(stock_id, industry_name)
                    feature_df.to_csv(feature_path, index=False, encoding=self.config.ENCODING)
                    
                    self.logger.info(f"成功生成 {industry_name} 的特徵檔案")
                    
                except Exception as e:
                    self.logger.error(f"處理 {industry_name} 時發生錯誤: {str(e)}")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"生成特徵時發生錯誤: {str(e)}")
            return False

    def _calculate_technical_indicators(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict:
        """計算技術指標"""
        try:
            features = {}
            
            # 確保輸入數據類型正確
            close = close.astype(np.float64)
            high = high.astype(np.float64)
            low = low.astype(np.float64)
            volume = volume.astype(np.float64)
            
            # 計算各種技術指標
            # 移動平均線
            for period in [5, 10, 20, 60]:
                ma = ta.SMA(close, timeperiod=period)
                if not np.isnan(ma[-1]):
                    features[f'MA{period}'] = float(ma[-1])
            
            # RSI
            rsi = ta.RSI(close, timeperiod=14)
            if not np.isnan(rsi[-1]):
                features['RSI'] = float(rsi[-1])
            
            # MACD
            macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            if not np.isnan(macd[-1]) and not np.isnan(signal[-1]) and not np.isnan(hist[-1]):
                features.update({
                    'MACD': float(macd[-1]),
                    'MACD_signal': float(signal[-1]),
                    'MACD_hist': float(hist[-1])
                })
            
            # 布林通道
            upper, middle, lower = ta.BBANDS(close, timeperiod=20)
            if not np.isnan(upper[-1]) and not np.isnan(middle[-1]) and not np.isnan(lower[-1]):
                features.update({
                    'BB_upper': float(upper[-1]),
                    'BB_middle': float(middle[-1]),
                    'BB_lower': float(lower[-1])
                })
            
            # KD指標
            slowk, slowd = ta.STOCH(high, low, close,
                                  fastk_period=9,
                                  slowk_period=3,
                                  slowk_matype=0,
                                  slowd_period=3,
                                  slowd_matype=0)
            if not np.isnan(slowk[-1]) and not np.isnan(slowd[-1]):
                features.update({
                    'K': float(slowk[-1]),
                    'D': float(slowd[-1])
                })
            
            # 成交量指標
            volume_ma5 = ta.SMA(volume, timeperiod=5)
            volume_ma20 = ta.SMA(volume, timeperiod=20)
            if not np.isnan(volume_ma5[-1]) and not np.isnan(volume_ma20[-1]):
                features.update({
                    'Volume_MA5': float(volume_ma5[-1]),
                    'Volume_MA20': float(volume_ma20[-1])
                })
            
            return features
            
        except Exception as e:
            self.logger.error(f"計算技術指標時發生錯誤: {str(e)}")
            return {}
    
    def _preprocess_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """數據預處理"""
        try:
            # 重命名欄位
            df = df.rename(columns=self.column_mapping)
            
            # 確保所有必要的欄位都存在
            required_columns = ['volume', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise KeyError(f'缺少必要欄位: {missing_columns}')
            
            # 處理日期欄位
            if '日期' in df.columns:
                df['日期'] = pd.to_datetime(df['日期'])
            
            # 將價格和成交量欄位轉換為數值型別
            numeric_columns = ['volume', 'open', 'high', 'low', 'close', 'amount', 'change', 'transactions']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 處理缺失值
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"數據預處理時發生錯誤: {str(e)}")
            return None
    
    def _load_technical_features(self, df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """讀取已有的技術指標"""
        try:
            # 使用配置文件中的路徑和文件名格式
            tech_file = (self.config.technical_analysis_path / 
                        f"{stock_id}_indicators.csv")
            
            if tech_file.exists():
                tech_df = pd.read_csv(tech_file, encoding=self.config.ENCODING)
                tech_df['日期'] = pd.to_datetime(tech_df['日期'])
                
                # 統一技術指標列名
                tech_df = tech_df.rename(columns=self.config.TECHNICAL_COLUMN_MAPPING)
                
                # 檢查必要的技術指標
                required_indicators = [
                    'RSI',               # RSI指標
                    'MACD', 'MACD_signal', 'MACD_hist',  # MACD系列
                    'slowk', 'slowd',    # KD指標
                    'TSF',               # 時間序列預測
                    'SAR',               # 拋物線指標
                    'middleband'         # 布林通道
                ]
                
                # 檢查移動平均線
                ma_types = self.config.TECHNICAL_PARAMS['ma']['types']
                ma_periods = self.config.TECHNICAL_PARAMS['ma']['periods']
                for ma_type in ma_types:
                    for period in ma_periods:
                        required_indicators.append(f"{ma_type}{period}")
                
                # 檢查哪些指標可用
                available_indicators = [col for col in required_indicators if col in tech_df.columns]
                missing_indicators = set(required_indicators) - set(available_indicators)
                
                if missing_indicators:
                    self.logger.warning(f"股票 {stock_id} 缺少以下技術指標: {missing_indicators}")
                
                # 合併技術指標到原始數據
                df = df.merge(tech_df[['日期'] + available_indicators], on='日期', how='left')
                
                # 處理缺失值
                df[available_indicators] = df[available_indicators].fillna(method='ffill').fillna(method='bfill')
                
                self.logger.info(f"已讀取股票 {stock_id} 的技術指標，可用指標: {available_indicators}")
                return df
            else:
                self.logger.warning(f"找不到股票 {stock_id} 的技術指標檔案: {tech_file}")
                return None
            
        except Exception as e:
            self.logger.error(f"讀取技術指標時發生錯誤: {str(e)}")
            return None
    
    def _add_industry_features(self, df: pd.DataFrame, stock_id: str) -> pd.DataFrame:
        """添加產業相關特徵"""
        try:
            # 獲取股票所屬產業
            stock_industry = None
            
            # 從產業對照檔案中查找
            if stock_id in self.industry_mapping['證券代號'].values:
                industry_info = self.industry_mapping[
                    self.industry_mapping['證券代號'] == stock_id
                ].iloc[0]
                stock_industry = industry_info['標準化產業']
            
            if stock_industry:
                # 讀取產業分析結果
                price_file = self.config.industry_analysis_path / 'price_index' / f'{stock_industry}_analysis.json'
                return_file = self.config.industry_analysis_path / 'return_index' / f'{stock_industry}_analysis.json'
                
                industry_data = pd.DataFrame()
                
                # 讀取價格指數分析
                if price_file.exists():
                    with open(price_file, 'r', encoding=self.config.ENCODING) as f:
                        price_data = json.load(f)
                        if price_data and 'daily_data' in price_data:
                            price_df = pd.DataFrame(price_data['daily_data'])
                            if not price_df.empty:
                                price_df['日期'] = pd.to_datetime(price_df['date'])
                                price_df['產業_價格趨勢'] = price_df['trend_direction']
                                price_df['產業_價格強度'] = price_df['trend_strength']
                                industry_data = price_df[['日期', '產業_價格趨勢', '產業_價格強度']]
                                self.logger.info(f"已讀取產業 {stock_industry} 的價格指數分析")
                
                # 讀取報酬指數分析
                if return_file.exists():
                    with open(return_file, 'r', encoding=self.config.ENCODING) as f:
                        return_data = json.load(f)
                        if return_data and 'daily_data' in return_data:
                            return_df = pd.DataFrame(return_data['daily_data'])
                            if not return_df.empty:
                                return_df['日期'] = pd.to_datetime(return_df['date'])
                                return_df['產業_報酬率'] = return_df['return_rate']
                                return_df['產業_波動率'] = return_df['volatility']
                                return_df['產業_動能得分'] = return_df['momentum_score']
                                return_df['產業_強度排名'] = return_df['strength_rank']
                                
                                if industry_data.empty:
                                    industry_data = return_df[['日期', '產業_報酬率', '產業_波動率', '產業_動能得分', '產業_強度排名']]
                                else:
                                    industry_data = industry_data.merge(
                                        return_df[['日期', '產業_報酬率', '產業_波動率', '產業_動能得分', '產業_強度排名']],
                                        on='日期',
                                        how='outer'
                                    )
                                self.logger.info(f"已讀取產業 {stock_industry} 的報酬指數分析")
                
                if not industry_data.empty:
                    # 合併產業特徵到原始數據
                    df = df.merge(industry_data, on='日期', how='left')
                    
                    # 處理缺失值
                    industry_columns = [
                        '產業_價格趨勢', '產業_價格強度', '產業_報酬率', 
                        '產業_波動率', '產業_動能得分', '產業_強度排名'
                    ]
                    df[industry_columns] = df[industry_columns].fillna(method='ffill').fillna(method='bfill')
                    
                    self.logger.info(f"已添加產業 {stock_industry} 的特徵")
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加產業特徵時發生錯誤: {str(e)}")
            return df
    
    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """後處理特徵"""
        try:
            # 處理缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 確保百分比指標在合理範圍內
            if 'RSI' in df.columns:
                df['RSI'] = np.clip(df['RSI'], 0, 100)
            if 'slowk' in df.columns:
                df['slowk'] = np.clip(df['slowk'], 0, 100)
            if 'slowd' in df.columns:
                df['slowd'] = np.clip(df['slowd'], 0, 100)
            
            return df
            
        except Exception as e:
            self.logger.error(f"後處理特徵時發生錯誤: {str(e)}")
            return df
    
    def _save_features(self, df: pd.DataFrame, stock_id: str) -> bool:
        """儲存特徵為CSV格式
        
        Args:
            df: 特徵數據框
            stock_id: 股票代碼
            
        Returns:
            bool: 是否成功儲存
        """
        try:
            # 確保日期欄位格式正確
            if '日期' in df.columns:
                if isinstance(df['日期'].iloc[0], str):
                    df['日期'] = pd.to_datetime(df['日期'])
            
            # 選擇要保存的特徵欄位
            feature_columns = [
                '日期', 'stock_id', 'stock_name',
                'open', 'high', 'low', 'close', 'volume',
                'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                'slowk', 'slowd',
                '產業_價格趨勢', '產業_價格強度', '產業_報酬率', 
                '產業_波動率', '產業_動能得分', '產業_強度排名'
            ]
            
            # 只保留存在的欄位
            existing_columns = [col for col in feature_columns if col in df.columns]
            output_df = df[existing_columns].copy()
            
            # 生成檔案名稱
            filename = self.config.get_feature_filename(stock_id)
            filepath = self.config.features_path / filename
            
            # 確保目錄存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 儲存為CSV
            output_df.to_csv(filepath, index=False, encoding=self.config.ENCODING)
            
            self.logger.info(f"已儲存特徵至: {filepath}")
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
    
    def _generate_processing_report(self, success_count: int, total_count: int):
        """生成處理報告"""
        try:
            report_path = self.config.meta_data_path / 'feature_generation_report.txt'
            
            with open(report_path, 'w', encoding=self.config.ENCODING) as f:
                f.write(f"特徵生成報告 - {datetime.now():%Y-%m-%d %H:%M:%S}\n")
                f.write("-" * 50 + "\n\n")
                f.write(f"處理總數: {total_count}\n")
                f.write(f"成功數量: {success_count}\n")
                f.write(f"失敗數量: {total_count - success_count}\n")
                f.write(f"成功率: {(success_count/total_count)*100:.2f}%\n\n")
                
                f.write("處理設定:\n")
                f.write(f"- 起始日期: {self.config.TEST_SETTING['start_date']}\n")
                f.write(f"- 結束日期: {self.config.TEST_SETTING['end_date']}\n")
                
            self.logger.info(f"處理報告已生成: {report_path}")
            
        except Exception as e:
            self.logger.error(f"生成處理報告時發生錯誤: {str(e)}") 