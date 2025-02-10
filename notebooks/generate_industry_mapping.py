import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum
import logging
import warnings
import sys
import json
import traceback
import shutil

# 設定警告忽略
warnings.filterwarnings('ignore')

class IndexType(Enum):
    """指數類型"""
    PRICE = "價格指數"    # 一般類指數
    RETURN = "報酬指數"   # 類報酬指數
    LEVERAGE = "槓桿指數"  # 兩倍槓桿指數
    INVERSE = "反向指數"   # 反向指數

class IndustryAnalysisSystem:
    """產業分析系統"""
    
    def __init__(self, base_path: str = ".."):
        self.base_path = Path(base_path).resolve()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 建立必要的目錄結構
        self._initialize_directories()
        
        try:
            # 載入基礎資料
            self.company_data = pd.read_csv(self.base_path / "meta_data/companies.csv")
            self.logger.info("成功載入公司資料")
            
            self.industry_index = pd.read_csv(self.base_path / "meta_data/industry_index.csv")
            self.logger.info("成功載入產業指數資料")
            
            # 建立產業對應關係
            self.industry_mapping = self._create_industry_mapping()
            
        except Exception as e:
            self.logger.error(f"初始化時發生錯誤: {str(e)}")
            raise
            
    def _initialize_directories(self):
        """初始化目錄結構"""
        directories = [
            "meta_data/backup",
            "industry_analysis/price_index",
            "industry_analysis/return_index"
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)
            
    def _create_industry_mapping(self) -> Dict:
        """建立產業分類對應關係"""
        mapping = {}
        
        # 建立指數類型分類
        def get_index_type(name: str) -> IndexType:
            """判斷指數類型"""
            if '報酬指數' in name:
                return IndexType.RETURN
            elif '兩倍槓桿指數' in name or '日報酬兩倍指數' in name:
                return IndexType.LEVERAGE
            elif '反向指數' in name or '反向一倍指數' in name:
                return IndexType.INVERSE
            else:
                return IndexType.PRICE
        
        index_types = {
            name: get_index_type(name)
            for name in self.industry_index['指數名稱'].unique()
        }
        
        # 產業名稱標準化
        def get_base_name(name: str) -> str:
            """獲取基礎產業名稱"""
            return name.replace('類報酬指數', '')\
                      .replace('類指數', '')\
                      .replace('類日報酬兩倍指數', '')\
                      .replace('類日報酬反向一倍指數', '')\
                      .replace('類兩倍槓桿指數', '')\
                      .replace('類反向指數', '')\
                      .strip()
        
        # 建立產業分類到指數的映射
        for index_name in self.industry_index['指數名稱'].unique():
            base_name = get_base_name(index_name)
            if base_name not in mapping:
                mapping[base_name] = {
                    IndexType.PRICE: [],
                    IndexType.RETURN: [],
                    IndexType.LEVERAGE: [],
                    IndexType.INVERSE: [],
                    'categories': set()
                }
            
            # 根據指數類型分類
            index_type = index_types[index_name]
            mapping[base_name][index_type].append(index_name)
        
        # 添加產業類別對應關係
        for category in self.company_data['industry_category'].unique():
            base_name = get_base_name(category)
            if base_name in mapping:
                mapping[base_name]['categories'].add(category)
        
        # 將set轉換回list
        for base_name in mapping:
            mapping[base_name]['categories'] = list(mapping[base_name]['categories'])
        
        return mapping
    
    def get_industry_stocks(self, industry_name: str) -> List[str]:
        """獲取特定產業的所有股票"""
        base_name = industry_name.replace('類報酬指數', '').replace('類指數', '').strip()
        all_stocks = set()
        
        if base_name in self.industry_mapping:
            # 獲取所有相關的產業類別
            categories = self.industry_mapping[base_name]['categories']
            
            # 對每個類別獲取股票
            for category in categories:
                stocks = self.company_data[
                    self.company_data['industry_category'] == category
                ]['stock_id'].unique()
                all_stocks.update(stocks)
        
        return list(all_stocks)
    
    def generate_industry_mapping_file(self, output_path: str = "meta_data/industry_mapping_analysis.csv"):
        """生成產業對照分析檔案
        
        Args:
            output_path: 輸出檔案路徑
        """
        try:
            # 準備資料
            mapping_data = []
            
            for base_name, indices in self.industry_mapping.items():
                # 獲取該產業的所有股票
                stocks = self.get_industry_stocks(base_name)
                
                # 獲取指數資訊
                price_indices = indices[IndexType.PRICE]
                return_indices = indices[IndexType.RETURN]
                leverage_indices = indices[IndexType.LEVERAGE]
                inverse_indices = indices[IndexType.INVERSE]
                
                # 建立對照資料
                mapping_data.append({
                    '標準化產業': base_name,
                    '對應產業指數': price_indices[0] if price_indices else None,
                    '對應報酬指數': return_indices[0] if return_indices else None,
                    '對應槓桿指數': leverage_indices[0] if leverage_indices else None,
                    '對應反向指數': inverse_indices[0] if inverse_indices else None,
                    '產業分析檔案': f"{base_name}",
                    '包含股票數': len(stocks),
                    '包含股票': ','.join(stocks) if stocks else '',
                    '產業類別': ','.join(indices['categories']) if indices['categories'] else '',
                    '更新時間': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            # 轉換為DataFrame並儲存
            df = pd.DataFrame(mapping_data)
            
            # 確保輸出目錄存在
            output_file = self.base_path / output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果已存在舊檔案，先備份
            if output_file.exists():
                backup_path = self.base_path / "meta_data/backup" / f"industry_mapping_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                shutil.copy2(output_file, backup_path)
                print(f"已備份原有對照表至: {backup_path}")
            
            # 儲存新檔案
            df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"產業對照表已生成: {output_file}")
            
            return df
            
        except Exception as e:
            print(f"生成產業對照表時發生錯誤: {str(e)}")
            raise

def main():
    # 設定日誌級別
    logging.getLogger().setLevel(logging.INFO)
    
    print("開始生成產業對照表...")
    
    try:
        # 初始化系統
        analyzer = IndustryAnalysisSystem()
        
        # 生成產業對照表
        mapping_df = analyzer.generate_industry_mapping_file()
        
        print("\n產業對照表生成成功！")
        print(f"\n產業對照表概要：")
        print(f"- 總產業數：{len(mapping_df)}")
        print(f"- 包含價格指數的產業數：{mapping_df['對應產業指數'].notna().sum()}")
        print(f"- 包含報酬指數的產業數：{mapping_df['對應報酬指數'].notna().sum()}")
        print(f"- 包含槓桿指數的產業數：{mapping_df['對應槓桿指數'].notna().sum()}")
        print(f"- 包含反向指數的產業數：{mapping_df['對應反向指數'].notna().sum()}")
        
        # 顯示前幾筆資料
        print("\n產業對照表預覽（前5筆）：")
        print(mapping_df[['標準化產業', '對應產業指數', '對應報酬指數', '包含股票數']].head())
        
    except Exception as e:
        print(f"生成產業對照表時發生錯誤：{str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 