import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import warnings
import sys
import time
import json
import traceback
warnings.filterwarnings('ignore')

class IndexType(Enum):
    """指數類型"""
    PRICE = "價格指數"    # 一般類指數
    RETURN = "報酬指數"   # 類報酬指數
    LEVERAGE = "槓桿指數"  # 兩倍槓桿指數
    INVERSE = "反向指數"   # 反向指數

class TimeFrequency(Enum):
    """時間頻率"""
    WEEKLY = 'W'
    MONTHLY = 'ME'

class IndustryAnalysisSystem:
    """產業分析系統"""
    
    def __init__(self, base_path: str = "D:/Min/Python/Project/FA_Data"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 建立必要的目錄結構
        self._initialize_directories()
        
        try:
            # 載入基礎資料
            self.company_data = pd.read_csv(self.base_path / "meta_data" / "companies.csv")
            self.logger.info("成功載入公司資料")
            
            self.industry_index = pd.read_csv(self.base_path / "meta_data" / "industry_index.csv")
            self.logger.info("成功載入產業指數資料")
            
            self.market_index = pd.read_csv(self.base_path / "meta_data" / "market_index.csv")
            self.logger.info("成功載入市場指數資料")
            
            # 處理日期格式
            self.industry_index['日期'] = pd.to_datetime(self.industry_index['日期'])
            self.market_index['Date'] = pd.to_datetime(self.market_index['Date'])
            
            # 建立產業對應關係
            self.industry_mapping = self._create_industry_mapping()
            
            # 驗證產業指數配對
            validation_results = self.validate_index_pairs()
            if validation_results['missing_pairs'] or validation_results['incomplete_pairs']:
                self.logger.warning("發現不完整的產業指數配對:")
                if validation_results['missing_pairs']:
                    self.logger.warning(f"缺少配對的產業: {validation_results['missing_pairs']}")
                if validation_results['incomplete_pairs']:
                    self.logger.warning(f"不完整配對的產業: {validation_results['incomplete_pairs']}")
                if validation_results['special_indices']:
                    self.logger.info(f"特殊指數產業: {validation_results['special_indices']}")
                    
        except FileNotFoundError as e:
            self.logger.error(f"找不到必要的資料檔案: {str(e)}")
            raise
        except pd.errors.EmptyDataError as e:
            self.logger.error(f"資料檔案是空的: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"初始化時發生未預期的錯誤: {str(e)}")
            raise

    def _initialize_directories(self):
        """初始化目錄結構"""
        directories = [
            "industry_correlation/weekly",
            "industry_correlation/monthly",
            "industry_analysis/price_index",
            "industry_analysis/return_index",
            "meta_data/backup"
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)

    def get_industry_stocks(self, industry_name: str) -> List[str]:
        """獲取特定產業的所有股票
        
        Args:
            industry_name: 產業名稱
            
        Returns:
            List[str]: 該產業的所有股票代碼列表
        """
        try:
            # 移除產業名稱中的指數類型標記
            base_name = industry_name.replace('類報酬指數', '')\
                                   .replace('類指數', '')\
                                   .replace('類日報酬兩倍指數', '')\
                                   .replace('類日報酬反向一倍指數', '')\
                                   .strip()
            
            # 從公司資料中篩選出屬於該產業的股票
            industry_stocks = self.company_data[
                self.company_data['industry_category'].str.contains(base_name, na=False)
            ]['stock_id'].unique().tolist()
            
            if not industry_stocks:
                self.logger.warning(f"找不到產業 {base_name} 的相關股票")
                
            return industry_stocks
            
        except Exception as e:
            self.logger.error(f"獲取產業股票時發生錯誤: {str(e)}")
            return []

    def validate_index_pairs(self) -> Dict:
        """驗證產業指數配對完整性"""
        validation_results = {
            'missing_pairs': [],      # 缺少配對的產業
            'incomplete_pairs': [],   # 不完整配對的產業
            'special_indices': []     # 有特殊指數的產業
        }
        
        for base_name, indices in self.industry_mapping.items():
            # 檢查基本配對
            has_price = bool(indices[IndexType.PRICE])
            has_return = bool(indices[IndexType.RETURN])
            has_leverage = bool(indices[IndexType.LEVERAGE])
            has_inverse = bool(indices[IndexType.INVERSE])
            
            # 完全缺少指數的產業
            if not any([has_price, has_return, has_leverage, has_inverse]):
                validation_results['missing_pairs'].append(base_name)
                continue
                
            # 檢查價格和報酬指數配對
            if has_price != has_return:
                validation_results['incomplete_pairs'].append({
                    'industry': base_name,
                    'has_price': has_price,
                    'has_return': has_return
                })
            
            # 檢查特殊指數
            if has_leverage or has_inverse:
                validation_results['special_indices'].append({
                    'industry': base_name,
                    'has_leverage': has_leverage,
                    'has_inverse': has_inverse
                })
        
        return validation_results
    
    def get_industry_performance(self, 
                               industry_name: str, 
                               index_type: IndexType = None,
                               start_date: str = None, 
                               end_date: str = None) -> pd.DataFrame:
        """獲取產業指數表現"""
        base_name = industry_name.replace('類報酬指數', '').replace('類指數', '').strip()
        
        if base_name in self.industry_mapping:
            # 根據指數類型選擇指數
            if index_type is None:
                # 如果未指定類型，優先使用報酬指數
                index_names = (self.industry_mapping[base_name][IndexType.RETURN] or 
                             self.industry_mapping[base_name][IndexType.PRICE])
            else:
                index_names = self.industry_mapping[base_name][index_type]
                
            if not index_names:
                return pd.DataFrame()
            
            # 獲取所有相關指數的數據
            industry_data = self.industry_index[
                self.industry_index['指數名稱'].isin(index_names)
            ].copy()
            
            # 日期過濾
            if start_date:
                industry_data = industry_data[
                    industry_data['日期'] >= pd.to_datetime(start_date)
                ]
            if end_date:
                industry_data = industry_data[
                    industry_data['日期'] <= pd.to_datetime(end_date)
                ]
            
            if not industry_data.empty:
                # 計算相關指標
                industry_data['daily_return'] = industry_data.groupby('指數名稱')['收盤指數'].pct_change()
                industry_data['volatility'] = industry_data.groupby('指數名稱')['daily_return'].rolling(20).std().values
            
            return industry_data
        
        return pd.DataFrame()
    
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
            index_type = index_types[index_name]
            
            if base_name not in mapping:
                mapping[base_name] = {
                    IndexType.PRICE: [],
                    IndexType.RETURN: [],
                    IndexType.LEVERAGE: [],
                    IndexType.INVERSE: [],
                    'categories': []
                }
            
            mapping[base_name][index_type].append(index_name)
        
        return mapping

    def generate_industry_analysis(self, start_date: str, end_date: str):
        """生成產業分析檔案"""
        try:
            # 遍歷所有產業
            for industry_name in self.industry_mapping.keys():
                # 獲取產業表現數據
                industry_data = self.get_industry_performance(
                    industry_name=industry_name,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if industry_data.empty:
                    self.logger.warning(f"無法獲取 {industry_name} 的產業數據")
                    continue
                
                # 獲取產業內的股票
                industry_stocks = self.get_industry_stocks(industry_name)
                
                # 計算時間序列分析
                time_series_analysis = self._calculate_time_series_analysis(industry_data)
                
                # 計算風險分析
                risk_analysis = self._calculate_risk_analysis(industry_data)
                
                # 計算輪動分析
                rotation_analysis = self._calculate_rotation_analysis(industry_data)
                
                # 生成投資建議
                investment_suggestions = self._generate_investment_suggestions(
                    time_series_analysis,
                    risk_analysis,
                    rotation_analysis
                )
                
                # 獲取資料的實際日期範圍
                data_start = industry_data['日期'].min().strftime('%Y%m%d')
                data_end = industry_data['日期'].max().strftime('%Y%m%d')
                current_date = '20250211'  # 使用固定的日期
                
                # 準備分析數據
                analysis_data = {
                    "basic_info": {
                        "industry_name": industry_name,
                        "period": {
                            "data_range": {
                                "start": data_start,
                                "end": data_end
                            },
                            "report_date": current_date
                        },
                        "stocks": industry_stocks
                    },
                    "time_series_analysis": time_series_analysis,
                    "risk_analysis": risk_analysis,
                    "rotation_analysis": rotation_analysis,
                    "investment_suggestions": investment_suggestions
                }
                
                # 儲存價格指數分析
                price_file = self.base_path / 'industry_analysis' / 'price_index' / f"{industry_name}_{data_start}_{data_end}_{current_date}.json"
                with open(price_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"成功生成 {industry_name} 的產業分析檔案")
                
        except Exception as e:
            self.logger.error(f"生成產業分析檔案時發生錯誤: {str(e)}")
            traceback.print_exc()

    def _calculate_time_series_analysis(self, data: pd.DataFrame) -> Dict:
        """計算時間序列分析"""
        try:
            # 準備價格數據
            prices = data['收盤指數'].values
            dates = data['日期'].values
            returns = data['daily_return'].values
            
            # 計算趨勢
            X = np.arange(len(prices)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X, prices)
            
            trend = {
                "slope": float(model.coef_[0]),
                "trend_direction": "上升" if model.coef_[0] > 0 else "下降",
                "trend_strength": abs(float(model.coef_[0]) / np.mean(prices)),
                "r2_score": float(model.score(X, prices)),
                "price_range": {
                    "start_price": float(prices[0]),
                    "end_price": float(prices[-1]),
                    "total_return": float((prices[-1] / prices[0] - 1) * 100)
                }
            }
            
            # 計算季節性
            dates_pd = pd.to_datetime(dates)
            monthly_returns = pd.DataFrame({
                'month': dates_pd.month,
                'return': returns
            }).groupby('month')['return'].agg(['mean', 'std'])
            
            quarterly_returns = pd.DataFrame({
                'quarter': dates_pd.quarter,
                'return': returns
            }).groupby('quarter')['return'].agg(['mean', 'std'])
            
            seasonality = {
                "monthly": {
                    "strongest_month": int(monthly_returns['mean'].idxmax()),
                    "weakest_month": int(monthly_returns['mean'].idxmin()),
                    "monthly_pattern": monthly_returns['mean'].to_dict(),
                    "monthly_volatility": monthly_returns['std'].to_dict()
                },
                "quarterly": {
                    "strongest_quarter": int(quarterly_returns['mean'].idxmax()),
                    "weakest_quarter": int(quarterly_returns['mean'].idxmin()),
                    "quarterly_pattern": quarterly_returns['mean'].to_dict(),
                    "quarterly_volatility": quarterly_returns['std'].to_dict()
                }
            }
            
            # 計算領先落後關係
            correlations = []
            max_lag = 10
            for lag in range(-max_lag, max_lag + 1):
                if lag == 0:
                    continue
                corr = np.corrcoef(prices[max_lag:-max_lag], 
                                 prices[max_lag+lag:len(prices)-max_lag+lag])[0, 1]
                correlations.append((lag, corr))
            
            max_corr = max(correlations, key=lambda x: abs(x[1]))
            
            lead_lag = {
                "max_correlation_lag": int(max_corr[0]),
                "max_correlation_value": float(max_corr[1]),
                "relationship_type": "領先" if max_corr[0] < 0 else "落後",
                "all_correlations": {str(lag): float(corr) for lag, corr in correlations}
            }
            
            return {
                "trend": trend,
                "seasonality": seasonality,
                "lead_lag": lead_lag
            }
            
        except Exception as e:
            self.logger.error(f"計算時間序列分析時發生錯誤: {str(e)}")
            return {}

    def _calculate_risk_analysis(self, data: pd.DataFrame) -> Dict:
        """計算風險分析"""
        try:
            returns = data['daily_return'].values
            prices = data['收盤指數'].values
            
            # 計算年化指標
            annual_return = np.mean(returns) * 252
            annual_volatility = np.std(returns) * np.sqrt(252)
            risk_free_rate = 0.02  # 假設無風險利率為2%
            sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
            
            # 計算下檔風險
            downside_returns = returns[returns < 0]
            downside_volatility = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
            
            # 計算回撤
            cumulative_returns = np.cumprod(1 + returns)
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = np.min(drawdowns)
            
            # 計算VaR和CVaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            return {
                "ratios": {
                    "annual_return": float(annual_return),
                    "annual_volatility": float(annual_volatility),
                    "sharpe_ratio": float(sharpe_ratio)
                },
                "downside": {
                    "downside_volatility": float(downside_volatility),
                    "sortino_ratio": float(sortino_ratio),
                    "loss_frequency": float(len(downside_returns) / len(returns)),
                    "avg_loss": float(np.mean(downside_returns))
                },
                "tail_risk": {
                    "var_95": float(var_95),
                    "var_99": float(var_99),
                    "cvar_95": float(cvar_95),
                    "cvar_99": float(cvar_99),
                    "skewness": float(stats.skew(returns)),
                    "kurtosis": float(stats.kurtosis(returns))
                },
                "drawdown": {
                    "max_drawdown": float(max_drawdown),
                    "avg_drawdown": float(np.mean(drawdowns)),
                    "avg_recovery_time": None,  # 需要額外計算
                    "max_recovery_time": None   # 需要額外計算
                }
            }
            
        except Exception as e:
            self.logger.error(f"計算風險分析時發生錯誤: {str(e)}")
            return {}

    def _calculate_rotation_analysis(self, data: pd.DataFrame) -> Dict:
        """計算輪動分析"""
        try:
            returns = data['daily_return'].values
            prices = data['收盤指數'].values
            volumes = data['成交金額'].values if '成交金額' in data.columns else None
            
            # 計算動能指標
            lookback_period = 120
            momentum_period = 20
            
            # 價格動能
            price_momentum = (prices[-1] / prices[-momentum_period] - 1) * 100
            
            # 成交量動能
            volume_momentum = 0
            if volumes is not None:
                volume_momentum = (np.mean(volumes[-momentum_period:]) / 
                                 np.mean(volumes[-lookback_period:-momentum_period]) - 1) * 100
            
            # 綜合動能分數
            composite_score = price_momentum * 0.7 + volume_momentum * 0.3
            
            return {
                "strength_ranking": {
                    "period_returns": float(np.sum(returns)),
                    "ranking": None,  # 需要與其他產業比較
                    "strong_industries": [],  # 需要與其他產業比較
                    "weak_industries": []     # 需要與其他產業比較
                },
                "momentum_ranking": {
                    "scores": {
                        "price_momentum": float(price_momentum),
                        "volume_momentum": float(volume_momentum),
                        "composite_score": float(composite_score)
                    },
                    "ranking": None  # 需要與其他產業比較
                },
                "flow_analysis": {
                    "indicators": {
                        "volume_change": float(volume_momentum),
                        "price_volume_ratio": float(price_momentum / volume_momentum) if volume_momentum != 0 else 0
                    },
                    "inflow_industries": [],  # 需要與其他產業比較
                    "outflow_industries": [], # 需要與其他產業比較
                    "rankings": {
                        "momentum": None,     # 需要與其他產業比較
                        "volume": None        # 需要與其他產業比較
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"計算輪動分析時發生錯誤: {str(e)}")
            return {}

    def _generate_investment_suggestions(self, 
                                      time_series_analysis: Dict,
                                      risk_analysis: Dict,
                                      rotation_analysis: Dict) -> Dict:
        """生成投資建議"""
        try:
            # 風險評估
            risk_level = "高"
            if (risk_analysis['ratios']['sharpe_ratio'] > 1 and 
                abs(risk_analysis['drawdown']['max_drawdown']) < 0.2):
                risk_level = "低"
            elif (risk_analysis['ratios']['sharpe_ratio'] > 0.5 and 
                  abs(risk_analysis['drawdown']['max_drawdown']) < 0.3):
                risk_level = "中"
            
            # 進場時機建議
            timing_suggestion = "觀望"
            if (time_series_analysis['trend']['trend_direction'] == "上升" and 
                rotation_analysis['momentum_ranking']['scores']['composite_score'] > 0):
                timing_suggestion = "適合進場"
            elif (time_series_analysis['trend']['trend_direction'] == "下降" and 
                  rotation_analysis['momentum_ranking']['scores']['composite_score'] < 0):
                timing_suggestion = "不適合進場"
            
            # 持倉建議
            position_suggestion = "中性"
            if timing_suggestion == "適合進場" and risk_level == "低":
                position_suggestion = "可適度加碼"
            elif timing_suggestion == "不適合進場" or risk_level == "高":
                position_suggestion = "建議減碼"
            
            return {
                "risk_assessment": risk_level,
                "timing_suggestions": timing_suggestion,
                "position_suggestions": position_suggestion,
                "key_points": [
                    f"夏普比率: {risk_analysis['ratios']['sharpe_ratio']:.2f}",
                    f"最大回撤: {risk_analysis['drawdown']['max_drawdown']*100:.2f}%",
                    f"產業趨勢: {time_series_analysis['trend']['trend_direction']}",
                    f"建議持倉水位: {position_suggestion}"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"生成投資建議時發生錯誤: {str(e)}")
            return {}

def main():
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 初始化產業分析系統
    system = IndustryAnalysisSystem()
    
    # 設定分析時間範圍
    start_date = "20230101"  # 與特徵生成的時間範圍一致
    end_date = "20241112"    # 與特徵生成的時間範圍一致
    
    # 生成產業分析檔案
    system.generate_industry_analysis(start_date, end_date)

if __name__ == "__main__":
    main() 