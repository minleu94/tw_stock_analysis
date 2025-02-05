import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import requests
import time
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Tuple
import yfinance as yf
from io import StringIO


class TWMarketDataProcessor:
    def __init__(self, config: Optional[TWStockConfig] = None, 
                 date_range: Optional[MarketDateRange] = None):
        """初始化數據處理器
        
        Args:
            config: TWStockConfig 實例，如果為 None 則創建新實例
            date_range: MarketDateRange 實例，如果為 None 則創建新實例
        """
        self.config = config or TWStockConfig()
        self.date_range = date_range or MarketDateRange()
        self.setup_logging()
        
        # 記錄設定的日期範圍
        self.logger.info(f"設定數據處理範圍: {self.date_range.date_range_str}")
    
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

    def get_latest_date(self, file_path: Path, date_column: str = '日期') -> Optional[str]:
        """獲取指定文件的最新日期"""
        if not file_path.exists():
            return None
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            return df[date_column].max()
        except Exception as e:
            self.logger.error(f"讀取{file_path}的最新日期時發生錯誤: {str(e)}")
            return None

    def get_daily_stock_data(self, date_str: str) -> Optional[pd.DataFrame]:
        """從TWSE獲取每日股票資料"""
        url = f'http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={date_str}&type=ALL'
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.warning(f"無法獲取 {date_str} 的數據: HTTP {response.status_code}")
                return None

            lines = [line.strip() for line in response.text.split('\n') if line.strip()]
            data_lines = []
            start_parsing = False
            
            for line in lines:
                if '證券代號' in line:
                    start_parsing = True
                    data_lines.append(line)
                elif start_parsing and not line.startswith('='):
                    data_lines.append(line)

            if data_lines:
                df = pd.read_csv(StringIO('\n'.join(data_lines)), thousands=',')
                df = df.loc[:, ~df.columns.str.contains("Unnamed")]
                
                # 標準化欄位名稱
                expected_columns = [
                    '證券代號', '證券名稱', '成交股數', '成交筆數', '成交金額',
                    '開盤價', '最高價', '最低價', '收盤價', '漲跌(+/-)',
                    '漲跌價差', '最後揭示買價', '最後揭示買量',
                    '最後揭示賣價', '最後揭示賣量', '本益比'
                ]
                
                # 確保所有必要欄位存在
                for col in expected_columns:
                    if col not in df.columns:
                        df[col] = None
                
                # 數據類型轉換
                df['證券代號'] = df['證券代號'].astype(str)
                numeric_columns = [
                    '成交股數', '成交筆數', '成交金額', '開盤價', '最高價',
                    '最低價', '收盤價', '漲跌價差', '最後揭示買價',
                    '最後揭示買量', '最後揭示賣價', '最後揭示賣量', '本益比'
                ]
                
                # 移除逗號並轉換為數值，處理 '--' 的情況
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = df[col].replace('--', np.nan)
                        df[col] = pd.to_numeric(
                            df[col].str.replace(',', '') if df[col].dtype == 'object' else df[col],
                            errors='coerce'
                        )
                
                return df
            return None

        except Exception as e:
            self.logger.error(f"處理 {date_str} 的股票數據時發生錯誤: {str(e)}")
            return None

    def process_daily_stock_data(self, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """處理每日股票數據"""
        try:
            # 確定日期範圍
            if start_date is None:
                latest_date = self.get_latest_date(self.config.stock_data_file)
                if latest_date:
                    start_date = (datetime.strptime(latest_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    start_date = '2014-01-01'  # 預設起始日期
            
            if end_date is None:
                end_date = datetime.today().strftime('%Y-%m-%d')

            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            self.logger.info(f"開始處理從 {start_date} 到 {end_date} 的每日股票數據")
            
            # 讀取現有數據
            existing_df = pd.DataFrame()
            if self.config.stock_data_file.exists():
                existing_df = pd.read_csv(self.config.stock_data_file, low_memory=False)
                existing_df['證券代號'] = existing_df['證券代號'].astype(str)
                self.logger.info(f"已讀取現有數據，共 {len(existing_df)} 筆記錄")

            # 收集新數據
            new_data_frames = []
            date_range = [start + timedelta(days=x) for x in range((end-start).days + 1)]
            
            for date in tqdm(date_range, desc="處理每日股票數據進度"):
                date_str = date.strftime('%Y%m%d')
                file_path = self.config.daily_price_dir / f'{date_str}.csv'
                
                if file_path.exists():
                    self.logger.debug(f"讀取現有文件 {date_str}")
                    daily_data = pd.read_csv(file_path)
                else:
                    daily_data = self.get_daily_stock_data(date_str)
                    if daily_data is not None:
                        daily_data.to_csv(file_path, index=False, encoding='utf-8-sig')
                        time.sleep(3)  # 避免過度頻繁請求
                
                if daily_data is not None:
                    daily_data['證券代號'] = daily_data['證券代號'].astype(str)
                    daily_data['日期'] = date.strftime('%Y-%m-%d')
                    new_data_frames.append(daily_data)

            if not new_data_frames:
                self.logger.info("沒有新的股票數據需要處理")
                return existing_df if not existing_df.empty else None

            # 合併新舊數據
            new_df = pd.concat(new_data_frames, ignore_index=True)
            if not existing_df.empty:
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['證券代號', '日期'], keep='last')
            else:
                df = new_df

            # 排序和保存
            df = df.sort_values(['證券代號', '日期'])
            df.to_csv(self.config.stock_data_file, index=False, encoding='utf-8-sig')
            
            self._generate_stock_report(df)
            return df

        except Exception as e:
            self.logger.error(f"處理每日股票數據時發生錯誤: {str(e)}")
            raise

    def update_market_index(self) -> Optional[pd.DataFrame]:
        """更新市場指數數據"""
        try:
            self.logger.info(f"開始更新市場指數數據: {self.date_range.date_range_str}")
            
            # 使用yfinance獲取TAIEX數據
            taiex_data = yf.download(
                "^TWII", 
                start=self.date_range.start_date, 
                end=self.date_range.end_date
            )
            
            if taiex_data.empty:
                self.logger.warning("未獲取到新的TAIEX數據")
                return None
            
            # 如果存在舊數據，進行合併
            if self.config.market_index_file.exists():
                old_data = pd.read_csv(
                    self.config.market_index_file, 
                    index_col='Date', 
                    parse_dates=True
                )
                taiex_data = pd.concat([old_data, taiex_data])
                taiex_data = taiex_data[~taiex_data.index.duplicated(keep='last')]
                taiex_data = taiex_data.sort_index()
            
            # 保存數據
            taiex_data.to_csv(self.config.market_index_file, encoding='utf-8-sig')
            self.logger.info(f"已更新TAIEX數據，共 {len(taiex_data)} 筆記錄")
            
            return taiex_data
            
        except Exception as e:
            self.logger.error(f"更新市場指數時發生錯誤: {str(e)}")
            raise

    def extract_index_data_for_date(self, date_str: str) -> Optional[List[Dict]]:
        """擷取特定日期的產業指數資料"""
        url = f'http://www.twse.com.tw/exchangeReport/MI_INDEX?response=csv&date={date_str}&type=ALL'
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.warning(f"無法獲取 {date_str} 的數據: HTTP {response.status_code}")
                return None

            lines = response.text.split('\n')
            index_data = []
            start_parsing = False
            
            for line in lines:
                if '類型' in line:
                    break
                    
                if start_parsing:
                    try:
                        parts = line.replace('"', '').strip().split(',')
                        if len(parts) >= 5 and any(x in parts[0] for x in ['指數', '發行量加權']):
                            index_data.append({
                                '指數名稱': parts[0].strip(),
                                '收盤指數': float(parts[1].replace(',', '')),
                                '漲跌': parts[2],
                                '漲跌點數': float(parts[3].replace(',', '')),
                                '漲跌百分比': float(parts[4].replace(',', '')),
                                '日期': datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
                            })
                    except (ValueError, IndexError):
                        continue
                
                if '收盤指數' in line:
                    start_parsing = True
                    
            return index_data
            
        except Exception as e:
            self.logger.error(f"處理 {date_str} 的指數數據時發生錯誤: {str(e)}")
            return None

    def process_industry_index_data(self) -> Optional[pd.DataFrame]:
        """處理產業指數數據"""
        try:
            self.logger.info(f"開始處理產業指數數據: {self.date_range.date_range_str}")
            
            # 讀取現有數據
            existing_df = pd.DataFrame()
            existing_dates = set()
            if self.config.industry_index_file.exists():
                existing_df = pd.read_csv(self.config.industry_index_file)
                existing_dates = set(existing_df['日期'].unique())
                self.logger.info(f"已讀取現有數據，共 {len(existing_df)} 筆記錄")
            
            # 生成需要處理的日期清單
            dates_to_process = []
            for date in self.date_range.get_date_list():
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    dates_to_process.append(date)
            
            if not dates_to_process:
                self.logger.info("所有日期的數據都已存在，無需更新")
                return existing_df
            
            self.logger.info(f"需要處理 {len(dates_to_process)} 天的數據")
            
            # 收集新數據
            new_data = []
            for date in tqdm(dates_to_process, desc="處理產業指數數據進度"):
                date_str = date.strftime('%Y%m%d')
                index_data = self.extract_index_data_for_date(date_str)
                
                if index_data:
                    new_data.extend(index_data)
                time.sleep(3)  # 避免請求過於頻繁
    
            if not new_data:
                self.logger.info("沒有新的產業指數數據需要處理")
                return existing_df if not existing_df.empty else None
    
            # 合併新舊數據
            new_df = pd.DataFrame(new_data)
            if not existing_df.empty:
                df = pd.concat([existing_df, new_df], ignore_index=True)
                df = df.drop_duplicates(subset=['日期', '指數名稱'], keep='last')
            else:
                df = new_df
    
            # 排序和保存
            df = df.sort_values(['指數名稱', '日期'])
            df.to_csv(self.config.industry_index_file, index=False, encoding='utf-8-sig')
            
            self._generate_index_report(df)
            return df
            
        except Exception as e:
            self.logger.error(f"處理產業指數數據時發生錯誤: {str(e)}")
            raise

    def _generate_stock_report(self, df: pd.DataFrame):
        """生成股票數據報告"""
        try:
            self.logger.info("\n=== 股票數據報告 ===")
            self.logger.info(f"總記錄數: {len(df):,d}")
            self.logger.info(f"股票數量: {len(df['證券代號'].unique()):,d}")
            self.logger.info(f"日期範圍: {df['日期'].min()} 到 {df['日期'].max()}")
            self.logger.info(f"總交易日數: {len(df['日期'].unique()):,d}")
            
            # 統計報告
            self._generate_period_statistics(df)
            self._generate_latest_day_statistics(df)
            
        except Exception as e:
            self.logger.error(f"生成股票報告時發生錯誤: {str(e)}")

    def _generate_period_statistics(self, df: pd.DataFrame):
        """生成期間統計報告"""
        try:
            self.logger.info(f"\n=== 期間統計 ({df['日期'].min()} 到 {df['日期'].max()}) ===")
            
            # 成交量排行
            period_volume = df.groupby(['證券代號', '證券名稱'])['成交股數'].sum().sort_values(ascending=False)
            self.logger.info("\n期間成交量最大的5支股票:")
            for idx, volume in period_volume.head().items():
                self.logger.info(f"  - {idx[0]} ({idx[1]}): {volume:,d} 股")
            
            # 成交金額排行
            period_value = df.groupby(['證券代號', '證券名稱'])['成交金額'].sum().sort_values(ascending=False)
            self.logger.info("\n期間成交金額最大的5支股票:")
            for idx, value in period_value.head().items():
                self.logger.info(f"  - {idx[0]} ({idx[1]}): NT$ {value:,.0f}")
                
        except Exception as e:
            self.logger.error(f"生成期間統計報告時發生錯誤: {str(e)}")

    def _generate_latest_day_statistics(self, df: pd.DataFrame):
        """生成最新交易日統計報告"""
        try:
            latest_date = df['日期'].max()
            latest_data = df[df['日期'] == latest_date]
            
            self.logger.info(f"\n=== 最新交易日統計 ({latest_date}) ===")
            self.logger.info(f"當日成交股票數: {len(latest_data):,d}")
            
            # 當日成交量排行
            daily_volume = latest_data.nlargest(5, '成交股數')
            self.logger.info("\n當日成交量最大的5支股票:")
            for _, row in daily_volume.iterrows():
                self.logger.info(f"  - {row['證券代號']} ({row['證券名稱']}): {row['成交股數']:,d} 股")
            
            # 當日成交金額排行
            daily_value = latest_data.nlargest(5, '成交金額')
            self.logger.info("\n當日成交金額最大的5支股票:")
            for _, row in daily_value.iterrows():
                self.logger.info(f"  - {row['證券代號']} ({row['證券名稱']}): NT$ {row['成交金額']:,.0f}")
            
            # 漲跌統計
            if '漲跌(+/-)' in latest_data.columns:
                up_count = len(latest_data[latest_data['漲跌(+/-)'] == '+'])
                down_count = len(latest_data[latest_data['漲跌(+/-)'] == '-'])
                unchanged = len(latest_data) - up_count - down_count
                
                self.logger.info(f"\n當日漲跌家數:")
                self.logger.info(f"  - 上漲: {up_count:,d}")
                self.logger.info(f"  - 下跌: {down_count:,d}")
                self.logger.info(f"  - 持平: {unchanged:,d}")
            
            # 最近交易日列表
            recent_dates = sorted(df['日期'].unique())[-5:]
            self.logger.info(f"\n最近的5個交易日: {', '.join(recent_dates)}")
            
        except Exception as e:
            self.logger.error(f"生成最新交易日統計報告時發生錯誤: {str(e)}")

    def process_all(self) -> Dict[str, pd.DataFrame]:
        """執行完整的數據處理流程"""
        self.logger.info(f"開始執行完整數據更新流程: {self.date_range.date_range_str}")
        
        results = {}
        try:
            # 1. 更新大盤指數
            self.logger.info("\n=== 更新大盤指數 ===")
            results['taiex_data'] = self.update_market_index()
            
            # 2. 更新產業指數
            self.logger.info("\n=== 更新產業指數 ===")
            results['industry_data'] = self.process_industry_index_data()
            
            # 3. 更新個股數據
            self.logger.info("\n=== 更新個股數據 ===")
            results['stock_data'] = self.process_daily_stock_data()
            
            # 4. 生成更新報告
            self._generate_update_summary()
            
            return results
            
        except Exception as e:
            self.logger.error(f"更新過程中發生錯誤: {str(e)}")
            raise

    def _generate_update_summary(self):
        """生成數據更新摘要報告"""
        self.logger.info("\n=== 數據更新摘要 ===")
        
        files_status = {
            '大盤指數': (self.config.market_index_file, 'Date'),
            '產業指數': (self.config.industry_index_file, '日期'),
            '個股數據': (self.config.stock_data_file, '日期')
        }
        
        for name, (file_path, date_col) in files_status.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    self.logger.info(f"{name}:")
                    self.logger.info(f"  - 資料筆數: {len(df):,d}")
                    self.logger.info(f"  - 日期範圍: {df[date_col].min()} 到 {df[date_col].max()}")
                    
                    if name == '產業指數':
                        self.logger.info(f"  - 指數數量: {len(df['指數名稱'].unique())}")
                    elif name == '個股數據':
                        self.logger.info(f"  - 股票數量: {len(df['證券代號'].unique())}")
                        
                except Exception as e:
                    self.logger.error(f"讀取 {name} 數據時發生錯誤: {str(e)}")
            else:
                self.logger.warning(f"{name} 數據文件不存在")