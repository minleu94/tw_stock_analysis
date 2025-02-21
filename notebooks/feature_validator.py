import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class FeatureValidator:
    """特徵驗證器"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 讀取產業對照檔案
        self.industry_mapping = pd.read_csv(
            self.config.meta_data_path / 'industry_mapping_analysis.csv'
        )
        
    def validate_features(self) -> bool:
        """執行完整的特徵驗證流程
        
        Returns:
            bool: 驗證是否全部通過
        """
        try:
            self.logger.info("開始特徵驗證流程...")
            
            # 1. 驗證文件結構
            self.logger.info("驗證文件結構...")
            existing_files, missing_files, industry_status = self._validate_file_structure()
            
            # 輸出驗證結果
            self._log_validation_results(existing_files, missing_files, industry_status)
            
            # 2. 驗證文件內容
            self.logger.info("驗證文件內容...")
            content_validation_passed = True
            for file in existing_files:
                if not self.validate_feature_content(file):
                    content_validation_passed = False
                    
            # 3. 驗證數據完整性
            self.logger.info("驗證數據完整性...")
            data_validation_passed = self._validate_data_completeness(existing_files)
            
            # 4. 合併特徵（如果需要）
            if content_validation_passed and data_validation_passed:
                self.logger.info("開始合併特徵文件...")
                if not self.combine_features(existing_files):
                    self.logger.warning("特徵合併失敗")
                    
            # 返回整體驗證結果
            validation_passed = (
                len(missing_files) == 0 and 
                content_validation_passed and 
                data_validation_passed
            )
            
            if validation_passed:
                self.logger.info("特徵驗證全部通過")
            else:
                self.logger.warning("特徵驗證發現問題")
                
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"驗證過程中發生錯誤: {str(e)}")
            return False
            
    def _validate_file_structure(self) -> Tuple[List[str], List[str], Dict[str, str]]:
        """驗證特徵文件結構
        
        Returns:
            Tuple[List[str], List[str], Dict[str, str]]: 
                - 存在的文件列表
                - 缺失的文件列表
                - 產業狀態字典
        """
        missing_files = []
        existing_files = []
        industry_status = {}
        
        # 檢查路徑是否存在
        if not os.path.exists(self.config.features_path):
            self.logger.error(f"特徵文件路徑不存在: {self.config.features_path}")
            return [], [], {}
            
        if not os.path.exists(self.config.industry_analysis_path):
            self.logger.error(f"產業分析資料路徑不存在: {self.config.industry_analysis_path}")
            return [], [], {}
            
        # 遍歷對照文件中的每個產業
        for _, row in self.industry_mapping.iterrows():
            industry_name = row['標準化產業']
            stock_id = row['對應產業指數']
            
            if pd.isna(stock_id) or stock_id == '未找到對應指數':
                industry_status[industry_name] = "無對應指數"
                continue
                
            # 檢查特徵文件
            feature_filename = self.config.get_feature_filename(stock_id, industry_name)
            feature_path = self.config.features_path / feature_filename
            
            if not feature_path.exists():
                missing_files.append(industry_name)
                industry_status[industry_name] = f"缺失特徵文件: {feature_filename}"
            else:
                existing_files.append(feature_filename)
                industry_status[industry_name] = "特徵文件存在"
                
        return existing_files, missing_files, industry_status
        
    def validate_feature_content(self, feature_file: str) -> bool:
        """驗證特徵文件內容
        
        Args:
            feature_file: 特徵文件名稱
            
        Returns:
            bool: 驗證是否通過
        """
        try:
            feature_path = self.config.features_path / feature_file
            
            # 檢查文件是否存在
            if not feature_path.exists():
                self.logger.error(f"特徵文件不存在: {feature_file}")
                return False
                
            # 讀取特徵文件
            features = pd.read_csv(feature_path, encoding=self.config.ENCODING)
            
            # 檢查必要的欄位
            required_columns = ['日期', 'open', 'high', 'low', 'close', 'volume', 'stock_id', 'stock_name']
            missing_columns = [col for col in required_columns if col not in features.columns]
            if missing_columns:
                self.logger.error(f"特徵文件缺少必要欄位: {missing_columns}")
                return False
            
            # 檢查日期格式
            features['日期'] = pd.to_datetime(features['日期'])
            
            # 檢查時間範圍
            if len(features) == 0:
                self.logger.error(f"特徵文件無資料: {feature_file}")
                return False
            
            # 檢查特徵資料的時間範圍
            start_date = pd.to_datetime(self.config.TEST_SETTING['start_date'])
            end_date = pd.to_datetime(self.config.TEST_SETTING['end_date'])
            data_start = features['日期'].min()
            data_end = features['日期'].max()
            
            if data_start > start_date or data_end < end_date:
                self.logger.error(f"特徵文件時間範圍不符: {feature_file}")
                self.logger.error(f"預期範圍: {start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}")
                self.logger.error(f"實際範圍: {data_start.strftime('%Y-%m-%d')} - {data_end.strftime('%Y-%m-%d')}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"驗證特徵文件內容時發生錯誤: {str(e)}")
            return False
            
    def _validate_data_completeness(self, existing_files: List[str]) -> bool:
        """驗證數據完整性
        
        Args:
            existing_files: 存在的特徵文件列表
            
        Returns:
            bool: 驗證是否通過
        """
        try:
            all_passed = True
            
            for file_name in existing_files:
                file_path = self.config.features_path / file_name
                features = pd.read_csv(file_path, encoding=self.config.ENCODING)
                
                # 檢查必要的特徵欄位
                required_columns = ['日期', 'open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in features.columns]
                if missing_columns:
                    self.logger.error(
                        f"文件 {file_name} 缺少必要欄位: {missing_columns}"
                    )
                    all_passed = False
                    continue
                
                # 檢查數值有效性
                for _, row in features.iterrows():
                    if not self._validate_price_logic(row):
                        self.logger.error(f"文件 {file_name} 在日期 {row['日期']} 的價格邏輯錯誤")
                        all_passed = False
                        
            return all_passed
            
        except Exception as e:
            self.logger.error(f"驗證數據完整性時發生錯誤: {str(e)}")
            return False
            
    def _validate_price_logic(self, row: pd.Series) -> bool:
        """驗證價格邏輯
        
        Args:
            row: 單日價格數據
            
        Returns:
            bool: 驗證是否通過
        """
        try:
            # 檢查價格是否為正數
            if any(row[field] <= 0 for field in ['open', 'high', 'low', 'close']):
                return False
                
            # 檢查最高價是否大於等於其他價格
            if not (row['high'] >= row['open'] and 
                   row['high'] >= row['low'] and 
                   row['high'] >= row['close']):
                return False
                
            # 檢查最低價是否小於等於其他價格
            if not (row['low'] <= row['open'] and 
                   row['low'] <= row['high'] and 
                   row['low'] <= row['close']):
                return False
                
            return True
            
        except Exception:
            return False
            
    def combine_features(self, existing_files: List[str]) -> bool:
        """合併特徵文件
        
        Args:
            existing_files: 存在的特徵文件列表
            
        Returns:
            bool: 合併是否成功
        """
        try:
            combined_features = pd.DataFrame()
            
            for file_name in existing_files:
                file_path = self.config.features_path / file_name
                try:
                    # 讀取特徵文件
                    features = pd.read_csv(file_path, encoding=self.config.ENCODING)
                    
                    # 添加產業名稱
                    industry_name = file_name.split('_')[0]
                    features['industry'] = industry_name
                    
                    # 合併數據
                    if combined_features.empty:
                        combined_features = features
                    else:
                        combined_features = pd.concat([combined_features, features], ignore_index=True)
                        
                except Exception as e:
                    self.logger.error(f"讀取文件 {file_name} 時發生錯誤: {str(e)}")
                    continue
            
            # 儲存合併後的特徵
            if not combined_features.empty:
                output_path = self.config.features_path / 'combined_features.csv'
                combined_features.to_csv(output_path, index=False, encoding=self.config.ENCODING)
                self.logger.info(f"特徵合併完成，已儲存至: {output_path}")
                return True
            else:
                self.logger.error("沒有可合併的特徵數據")
                return False
            
        except Exception as e:
            self.logger.error(f"合併特徵文件時發生錯誤: {str(e)}")
            return False
            
    def _log_validation_results(self, existing_files: List[str], missing_files: List[str], industry_status: Dict[str, str]) -> None:
        """記錄驗證結果
        
        Args:
            existing_files: 存在的文件列表
            missing_files: 缺失的文件列表
            industry_status: 產業狀態字典
        """
        report_path = self.config.meta_data_path / 'feature_validation_report.txt'
        
        with open(report_path, 'w', encoding=self.config.ENCODING) as f:
            f.write("=== 特徵驗證報告 ===\n\n")
            
            # 記錄時間
            f.write(f"驗證時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 記錄文件統計
            f.write("文件統計:\n")
            f.write(f"- 存在的文件數量: {len(existing_files)}\n")
            f.write(f"- 缺失的文件數量: {len(missing_files)}\n\n")
            
            # 記錄產業狀態
            f.write("產業狀態:\n")
            for industry, status in industry_status.items():
                f.write(f"- {industry}: {status}\n")
            f.write("\n")
            
            # 記錄存在的文件
            if existing_files:
                f.write("存在的文件:\n")
                for file in existing_files:
                    f.write(f"- {file}\n")
                f.write("\n")
            
            # 記錄缺失的文件
            if missing_files:
                f.write("缺失的文件:\n")
                for file in missing_files:
                    f.write(f"- {file}\n")
                f.write("\n") 