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
            analysis_file = row['產業分析檔案']
            
            if pd.isna(stock_id) or stock_id == '未找到對應指數':
                industry_status[industry_name] = "無對應指數"
                continue
                
            # 檢查特徵文件
            feature_filename = self.config.get_feature_filename(stock_id, analysis_file)
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
            with open(feature_path, 'r', encoding='utf-8') as f:
                features = json.load(f)
                
            # 檢查特徵內容
            if not isinstance(features, dict):
                self.logger.error(f"特徵文件格式錯誤: {feature_file}")
                return False
                
            # 檢查時間範圍
            dates = list(features.keys())
            if not dates:
                self.logger.error(f"特徵文件無資料: {feature_file}")
                return False
                
            # 解析文件名中的時間範圍
            file_parts = feature_file.split('_')
            if len(file_parts) >= 4:
                expected_start = file_parts[-2]
                expected_end = file_parts[-1].replace('.json', '')
                
                # 檢查特徵資料的時間範圍
                start_date = min(dates)
                end_date = max(dates)
                
                if start_date > expected_start or end_date < expected_end:
                    self.logger.error(f"特徵文件時間範圍不符: {feature_file}")
                    self.logger.error(f"預期範圍: {expected_start} - {expected_end}")
                    self.logger.error(f"實際範圍: {start_date} - {end_date}")
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                    
                # 檢查必要的特徵欄位
                required_fields = ['open', 'high', 'low', 'close', 'volume']
                for date, data in features.items():
                    missing_fields = [field for field in required_fields if field not in data]
                    if missing_fields:
                        self.logger.error(
                            f"文件 {file_name} 在日期 {date} 缺少必要欄位: {missing_fields}"
                        )
                        all_passed = False
                        
                # 檢查數值有效性
                for date, data in features.items():
                    if not self._validate_price_logic(data):
                        self.logger.error(f"文件 {file_name} 在日期 {date} 的價格邏輯錯誤")
                        all_passed = False
                        
            return all_passed
            
        except Exception as e:
            self.logger.error(f"驗證數據完整性時發生錯誤: {str(e)}")
            return False
            
    def _validate_price_logic(self, data: Dict) -> bool:
        """驗證價格邏輯
        
        Args:
            data: 單日價格數據
            
        Returns:
            bool: 驗證是否通過
        """
        try:
            # 檢查價格是否為正數
            if any(data[field] <= 0 for field in ['open', 'high', 'low', 'close']):
                return False
                
            # 檢查最高價是否大於等於其他價格
            if not (data['high'] >= data['open'] and 
                   data['high'] >= data['low'] and 
                   data['high'] >= data['close']):
                return False
                
            # 檢查最低價是否小於等於其他價格
            if not (data['low'] <= data['open'] and 
                   data['low'] <= data['high'] and 
                   data['low'] <= data['close']):
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
            combined_features = {}
            
            for file_name in existing_files:
                file_path = self.config.features_path / file_name
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        features = json.load(f)
                        industry_name = file_name.split('_')[0]
                        combined_features[industry_name] = features
                except Exception as e:
                    self.logger.error(f"讀取文件 {file_name} 時發生錯誤: {str(e)}")
                    continue
            
            # 儲存合併後的特徵
            output_path = self.config.features_path / 'combined_features.json'
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(combined_features, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"特徵合併完成，已儲存至: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"合併特徵文件時發生錯誤: {str(e)}")
            return False
            
    def _log_validation_results(self, 
                             existing_files: List[str], 
                             missing_files: List[str], 
                             industry_status: Dict[str, str]):
        """記錄驗證結果
        
        Args:
            existing_files: 存在的文件列表
            missing_files: 缺失的文件列表
            industry_status: 產業狀態字典
        """
        self.logger.info("\n產業特徵文件狀態報告：")
        self.logger.info("-" * 50)
        for industry, status in industry_status.items():
            self.logger.info(f"{industry}: {status}")
            
        self.logger.info("\n詳細的特徵文件列表：")
        self.logger.info("-" * 50)
        for file in existing_files:
            self.logger.info(f"- {file}")
            
        self.logger.info("\n缺失的產業特徵：")
        self.logger.info("-" * 50)
        for industry in missing_files:
            self.logger.info(f"- {industry}") 