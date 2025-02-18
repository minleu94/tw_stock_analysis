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
        """根據產業名稱獲取對應的檔案名稱"""
        # 常見的產業後綴
        suffixes = ['工業', '業']
        
        # 檢查是否已經包含後綴
        for suffix in suffixes:
            if industry_name.endswith(suffix):
                return industry_name
        
        # 根據特定規則添加後綴
        if industry_name in ['水泥', '食品', '塑膠', '紡織纖維', '電機機械', '電器電纜', '化學', '生技醫療',
                           '玻璃陶瓷', '造紙', '鋼鐵', '橡膠', '汽車', '半導體', '電腦及週邊設備', '光電',
                           '通信網路', '電子零組件', '電子通路', '資訊服務', '其他電子']:
            return f"{industry_name}工業"
        elif industry_name in ['建材營造', '航運', '觀光餐旅', '金融保險', '貿易百貨', '油電燃氣',
                             '居家生活', '數位雲端', '運動休閒', '綠能環保']:
            return f"{industry_name}業"
        
        return industry_name

    def generate_features(self) -> bool:
        """生成特徵"""
        try:
            self.logger.info("開始生成特徵...")
            
            # 遍歷產業對照檔案
            for _, row in self.industry_mapping.iterrows():
                industry_name = row['標準化產業']
                stock_id = row['對應產業指數']
                analysis_file = row['產業分析檔案']
                
                if pd.isna(stock_id) or stock_id == '未找到對應指數':
                    self.logger.info(f"{industry_name}: 無對應指數，跳過處理")
                    continue
                    
                try:
                    # 讀取產業分析資料
                    start_date = self.config.TEST_SETTING['start_date'].replace('-', '')  # 轉換日期格式為YYYYMMDD
                    end_date = self.config.TEST_SETTING['end_date'].replace('-', '')      # 轉換日期格式為YYYYMMDD
                    current_date = '20250211'  # 使用固定的日期
                    
                    # 使用產業分析檔案名稱
                    analysis_file_name = row['產業分析檔案']
                    if pd.isna(analysis_file_name) or not analysis_file_name:
                        self.logger.error(f"{industry_name}: 無產業分析檔案名稱")
                        continue
                        
                    # 構建檔案路徑
                    analysis_path = self.config.industry_analysis_path / 'price_index' / f"{industry_name}_{start_date}_{end_date}_{current_date}.json"
                    
                    if not analysis_path.exists():
                        # 嘗試使用不同的日期格式
                        start_date_alt = "20230103"  # 實際檔案中的開始日期
                        end_date_alt = "20241112"    # 實際檔案中的結束日期
                        
                        # 嘗試不同的產業名稱格式
                        possible_names = [
                            industry_name,
                            industry_name.replace('工業', ''),
                            industry_name.replace('業', ''),
                            industry_name + '工業',
                            industry_name + '業'
                        ]
                        
                        found = False
                        for name in possible_names:
                            test_path = self.config.industry_analysis_path / 'price_index' / f"{name}_{start_date_alt}_{end_date_alt}_{current_date}.json"
                            if test_path.exists():
                                analysis_path = test_path
                                found = True
                                break
                                
                        if not found:
                            self.logger.error(f"找不到產業分析檔案: {analysis_path}")
                            continue
                        
                    with open(analysis_path, 'r', encoding=self.config.ENCODING) as f:
                        industry_data = json.load(f)
                        
                    # 檢查數據結構
                    if 'basic_info' not in industry_data or 'time_series_analysis' not in industry_data:
                        self.logger.error(f"產業分析檔案格式不正確: {analysis_path}")
                        continue
                        
                    # 從時間序列分析中提取價格數據
                    time_series = industry_data['time_series_analysis']['price_series']
                    
                    # 轉換為DataFrame
                    df = pd.DataFrame(time_series).T
                    df.index = pd.to_datetime(df.index)
                    
                    # 重命名列
                    df.columns = ['開盤價', '最高價', '最低價', '收盤價', '成交量']
                    
                    if df.empty:
                        self.logger.error("轉換後的數據為空")
                        continue
                        
                    df = df.sort_index()
                    
                    # 篩選時間範圍
                    start_date = pd.to_datetime(self.config.TEST_SETTING['start_date'])
                    end_date = pd.to_datetime(self.config.TEST_SETTING['end_date'])
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df = df[mask]
                    
                    if len(df) < self.config.INDUSTRY_PARAMS['min_data_days']:
                        self.logger.warning(f"{industry_name}: 資料天數不足，跳過處理")
                        continue
                        
                    # 計算特徵
                    features = self._calculate_features(df)
                    
                    # 轉換特徵為DataFrame格式
                    feature_df = pd.DataFrame.from_dict(features, orient='index')
                    feature_df.index.name = '日期'
                    feature_df.reset_index(inplace=True)
                    feature_df['stock_id'] = stock_id
                    feature_df['stock_name'] = industry_name
                    
                    # 儲存特徵
                    feature_path = self.config.features_path / self.config.get_feature_filename(stock_id)
                    feature_df.to_csv(feature_path, index=False, encoding=self.config.ENCODING)
                    
                    self.logger.info(f"成功生成 {industry_name} 的特徵檔案")
                    
                except Exception as e:
                    self.logger.error(f"處理 {industry_name} 時發生錯誤: {str(e)}")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"生成特徵時發生錯誤: {str(e)}")
            return False

    def _calculate_features(self, df: pd.DataFrame) -> Dict:
        """計算技術指標特徵"""
        features = {}
        
        for date, row in df.iterrows():
            date_str = date.strftime('%Y%m%d')
            
            try:
                # 基本價格特徵
                price_features = {
                    'open': float(row['開盤價']),
                    'high': float(row['最高價']),
                    'low': float(row['最低價']),
                    'close': float(row['收盤價']),
                    'volume': float(row['成交量'])
                }
                
                # 計算技術指標
                tech_features = self._calculate_technical_indicators(df, date)
                
                # 合併特徵
                features[date_str] = {
                    **price_features,
                    **tech_features
                }
                
            except Exception as e:
                self.logger.error(f"計算 {date_str} 的特徵時發生錯誤: {str(e)}")
                continue
                
        return features

    def _calculate_technical_indicators(self, df: pd.DataFrame, current_date: datetime) -> Dict:
        """計算技術指標"""
        try:
            # 準備資料
            prices = df[df.index <= current_date]
            if len(prices) < 30:  # 確保有足夠的資料計算指標
                return {}
                
            close = prices['收盤價'].values
            high = prices['最高價'].values
            low = prices['最低價'].values
            volume = prices['成交量'].values
            
            # 計算各種技術指標
            features = {}
            
            # 移動平均線
            for period in [5, 10, 20, 60]:
                ma = ta.SMA(close, timeperiod=period)
                features[f'MA{period}'] = float(ma[-1])
                
            # RSI
            rsi = ta.RSI(close, timeperiod=14)
            features['RSI'] = float(rsi[-1])
            
            # MACD
            macd, signal, hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features.update({
                'MACD': float(macd[-1]),
                'MACD_signal': float(signal[-1]),
                'MACD_hist': float(hist[-1])
            })
            
            # 布林通道
            upper, middle, lower = ta.BBANDS(close, timeperiod=20)
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
            features.update({
                'K': float(slowk[-1]),
                'D': float(slowd[-1])
            })
            
            # 成交量指標
            features.update({
                'Volume_MA5': float(ta.SMA(volume, timeperiod=5)[-1]),
                'Volume_MA20': float(ta.SMA(volume, timeperiod=20)[-1])
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