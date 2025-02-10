import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import talib

class FeatureGenerator:
    """特徵生成器"""
    
    # 定義編碼常數
    ENCODING = 'utf-8'
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 讀取產業對照檔案
        self.industry_mapping = pd.read_csv(
            self.config.meta_data_path / 'industry_mapping_analysis.csv',
            encoding=self.ENCODING
        )
        
    def generate_features(self):
        """生成所有產業的特徵"""
        try:
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
                    analysis_path = self.config.industry_analysis_path / 'price_index' / f"{analysis_file}_20230101_20250122_20250122.json"
                    if not analysis_path.exists():
                        # 嘗試使用標準化產業名稱
                        analysis_path = self.config.industry_analysis_path / 'price_index' / f"{industry_name}_20230101_20250122_20250122.json"
                        if not analysis_path.exists():
                            self.logger.error(f"找不到產業分析檔案: {analysis_path}")
                            continue
                        
                    with open(analysis_path, 'r', encoding=self.ENCODING) as f:
                        industry_data = json.load(f)
                        
                    # 轉換為DataFrame
                    df = pd.DataFrame.from_dict(industry_data, orient='index')
                    # 確保索引是字符串格式
                    df.index = df.index.astype(str)
                    # 使用特定格式解析日期
                    df.index = pd.to_datetime(df.index, format='%Y%m%d')
                    df = df.sort_index()
                    
                    # 篩選時間範圍
                    mask = (df.index >= self.config.TEST_SETTING['start_date']) & \
                           (df.index <= self.config.TEST_SETTING['end_date'])
                    df = df[mask]
                    
                    if len(df) < self.config.INDUSTRY_PARAMS['min_data_days']:
                        self.logger.warning(f"{industry_name}: 資料天數不足，跳過處理")
                        continue
                        
                    # 計算技術指標
                    features = self._calculate_features(df)
                    
                    # 儲存特徵
                    feature_filename = self.config.get_feature_filename(stock_id, analysis_file)
                    feature_path = self.config.features_path / feature_filename
                    
                    with open(feature_path, 'w', encoding=self.ENCODING) as f:
                        json.dump(features, f, ensure_ascii=False, indent=2)
                        
                    self.logger.info(f"成功生成 {industry_name} 的特徵檔案")
                    
                except Exception as e:
                    self.logger.error(f"處理 {industry_name} 時發生錯誤: {str(e)}")
                    continue
                    
            return True
            
        except Exception as e:
            self.logger.error(f"生成特徵時發生錯誤: {str(e)}")
            return False
            
    def _calculate_features(self, df: pd.DataFrame) -> Dict:
        """計算技術指標特徵
        
        Args:
            df: 產業指數資料DataFrame
            
        Returns:
            Dict: 特徵字典
        """
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
        """計算技術指標
        
        Args:
            df: 產業指數資料DataFrame
            current_date: 當前日期
            
        Returns:
            Dict: 技術指標字典
        """
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
                ma = talib.SMA(close, timeperiod=period)
                features[f'MA{period}'] = float(ma[-1])
                
            # RSI
            rsi = talib.RSI(close, timeperiod=14)
            features['RSI'] = float(rsi[-1])
            
            # MACD
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            features.update({
                'MACD': float(macd[-1]),
                'MACD_signal': float(signal[-1]),
                'MACD_hist': float(hist[-1])
            })
            
            # 布林通道
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            features.update({
                'BB_upper': float(upper[-1]),
                'BB_middle': float(middle[-1]),
                'BB_lower': float(lower[-1])
            })
            
            # KD指標
            slowk, slowd = talib.STOCH(high, low, close, 
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
                'Volume_MA5': float(talib.SMA(volume, timeperiod=5)[-1]),
                'Volume_MA20': float(talib.SMA(volume, timeperiod=20)[-1])
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"計算技術指標時發生錯誤: {str(e)}")
            return {} 