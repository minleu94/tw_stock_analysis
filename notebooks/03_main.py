import logging
from pathlib import Path
from datetime import datetime
import sys
import os
import time
import pandas as pd
from feature_config import FeatureConfig
from feature_manager import FeatureManager
from feature_validator import FeatureValidator

def setup_logging(log_path: Path) -> logging.Logger:
    """設定日誌系統"""
    # 確保日誌目錄存在
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 創建logger
    logger = logging.getLogger("FeatureGeneration")
    logger.setLevel(logging.INFO)
    
    # 清除現有的處理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 創建文件處理器
    log_file = log_path / f'feature_generation_{datetime.now():%Y%m%d}.log'
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 創建控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 創建格式器
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def validate_config(config: FeatureConfig, logger: logging.Logger) -> bool:
    """驗證配置是否有效"""
    try:
        if not config.TEST_SETTING.get('start_date') or not config.TEST_SETTING.get('end_date'):
            logger.error("缺少必要的日期配置")
            return False
        if not config.TEST_SETTING.get('test_stocks'):
            logger.error("缺少測試股票列表")
            return False
        
        # 添加更多驗證
        if not isinstance(config.TEST_SETTING.get('test_stocks'), list):
            logger.error("test_stocks 必須是列表格式")
            return False
            
        # 驗證日期格式
        try:
            datetime.strptime(config.TEST_SETTING['start_date'], '%Y-%m-%d')
            datetime.strptime(config.TEST_SETTING['end_date'], '%Y-%m-%d')
        except ValueError:
            logger.error("日期格式錯誤，應為 YYYY-MM-DD")
            return False
            
        # 驗證開始日期小於結束日期
        if config.TEST_SETTING['start_date'] > config.TEST_SETTING['end_date']:
            logger.error("開始日期不能大於結束日期")
            return False
            
        return True
    except Exception as e:
        logger.error(f"配置驗證失敗: {str(e)}")
        return False

def check_features(config: FeatureConfig, logger: logging.Logger, stock_ids=None):
    """檢查特徵檔案
    
    Args:
        config: 配置對象
        logger: 日誌對象
        stock_ids: 要檢查的股票代碼列表，如果為None則檢查所有股票
    """
    try:
        logger.info("開始檢查特徵檔案...")
        
        # 設定特徵檔案路徑
        features_dir = config.features_path
        
        # 查找特徵檔案
        files = list(features_dir.glob("*.csv"))
        
        # 如果指定了股票代碼，則只檢查這些股票的特徵檔案
        if stock_ids:
            files = [f for f in files if any(stock_id in f.name for stock_id in stock_ids)]
        
        # 按修改時間排序
        files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not files:
            logger.warning("找不到特徵檔案")
            return
        
        # 設定顯示選項
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.max_rows', 20)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        # 讀取特徵檔案
        for file in files[:4]:  # 只讀取最新的四個檔案
            logger.info(f"檢查檔案: {file.name}")
            try:
                df = pd.read_csv(file)
                
                # 顯示基本資訊
                logger.info(f"資料形狀: {df.shape}")
                
                # 檢查日期欄位名稱
                date_col = None
                for col in ['date', '日期', 'Date', 'DATE']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    logger.info(f"資料日期範圍: {df[date_col].min()} 到 {df[date_col].max()}")
                else:
                    logger.warning("找不到日期欄位")
                
                # 檢查股票代碼和名稱欄位
                stock_id_col = None
                for col in ['stock_id', '股票代碼', 'StockID', 'STOCK_ID']:
                    if col in df.columns:
                        stock_id_col = col
                        break
                
                stock_name_col = None
                for col in ['stock_name', '股票名稱', 'StockName', 'STOCK_NAME']:
                    if col in df.columns:
                        stock_name_col = col
                        break
                
                if stock_id_col:
                    logger.info(f"股票代碼: {df[stock_id_col].iloc[0]}")
                else:
                    logger.warning("找不到股票代碼欄位")
                
                if stock_name_col:
                    logger.info(f"股票名稱: {df[stock_name_col].iloc[0]}")
                else:
                    logger.warning("找不到股票名稱欄位")
                
                # 顯示欄位數量
                all_columns = df.columns.tolist()
                logger.info(f"總共有 {len(all_columns)} 個欄位")
                
                # 檢查技術指標欄位
                tech_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'slowk', 'slowd', 'SAR', 'TSF', 'middleband', 'SMA30', 'EMA30', 'DEMA30']
                available_tech_cols = [col for col in tech_cols if col in df.columns]
                if available_tech_cols:
                    logger.info(f"找到 {len(available_tech_cols)} 個技術指標欄位")
                
                # 檢查產業相關欄位
                industry_cols = [col for col in df.columns if '產業_' in col or 'industry_' in col]
                if industry_cols:
                    logger.info(f"找到 {len(industry_cols)} 個產業相關欄位")
                
            except Exception as e:
                logger.error(f"讀取檔案 {file.name} 時發生錯誤: {str(e)}")
        
        logger.info("特徵檔案檢查完成")
        
    except Exception as e:
        logger.error(f"檢查特徵檔案時發生錯誤: {str(e)}")

def test_delete_files(config: FeatureConfig, logger: logging.Logger, stock_id, industry, start_date, end_date, auto_confirm=False):
    """測試刪除檔案功能
    
    Args:
        config: 配置對象
        logger: 日誌對象
        stock_id: 股票代碼
        industry: 產業名稱
        start_date: 開始日期 (格式: YYYYMMDD)
        end_date: 結束日期 (格式: YYYYMMDD)
        auto_confirm: 是否自動確認刪除檔案
    """
    try:
        logger.info("開始測試檔案刪除功能...")
        
        # 設定特徵檔案路徑
        features_dir = config.features_path
        
        # 使用更精確的模式匹配，匹配股票代碼、產業和日期範圍
        pattern = f"{stock_id}_{industry}_{start_date}_{end_date}_*.csv"
        logger.info(f"搜尋檔案模式: {pattern}")
        existing_files = list(features_dir.glob(pattern))
        logger.info(f"找到 {len(existing_files)} 個檔案")
        
        # 顯示找到的檔案
        for i, file in enumerate(existing_files):
            logger.info(f"{i+1}. {file.name}")
        
        # 如果找到多個檔案，則需要刪除舊檔案
        if len(existing_files) > 1:
            # 按修改時間排序
            existing_files = sorted(existing_files, key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 最新的檔案
            newest_file = existing_files[0]
            logger.info(f"最新的檔案: {newest_file.name}")
            
            # 需要刪除的舊檔案
            files_to_delete = existing_files[1:]
            logger.info(f"找到 {len(files_to_delete)} 個需要刪除的舊檔案:")
            for i, file in enumerate(files_to_delete):
                logger.info(f"{i+1}. {file.name}")
            
            # 詢問是否刪除檔案
            if files_to_delete:
                if auto_confirm:
                    choice = 'y'
                    logger.info("自動確認刪除檔案")
                else:
                    choice = input("\n是否刪除這些檔案？(y/n): ")
                
                if choice.lower() == 'y':
                    for old_file in files_to_delete:
                        try:
                            logger.info(f"嘗試刪除檔案: {old_file}")
                            if old_file.exists():
                                try:
                                    os.chmod(str(old_file), 0o777)  # 嘗試更改權限
                                    logger.info(f"已更改檔案權限: {old_file}")
                                except Exception as e:
                                    logger.error(f"更改檔案權限時發生錯誤: {str(e)}")
                                
                                deleted = False
                                
                                # 方法1: 使用pathlib的unlink
                                try:
                                    old_file.unlink()
                                    logger.info(f"成功使用unlink刪除檔案: {old_file}")
                                    deleted = True
                                except Exception as e:
                                    logger.error(f"使用unlink刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                                
                                # 方法2: 使用os.remove
                                if not deleted:
                                    try:
                                        os.remove(str(old_file))
                                        logger.info(f"使用os.remove成功刪除檔案: {old_file}")
                                        deleted = True
                                    except Exception as e:
                                        logger.error(f"使用os.remove刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                                
                                # 方法3: 使用os.unlink
                                if not deleted:
                                    try:
                                        os.unlink(str(old_file))
                                        logger.info(f"使用os.unlink成功刪除檔案: {old_file}")
                                        deleted = True
                                    except Exception as e:
                                        logger.error(f"使用os.unlink刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                                
                                # 檢查是否成功刪除
                                if not deleted:
                                    logger.warning(f"無法刪除檔案: {old_file}")
                            else:
                                logger.warning(f"檔案不存在: {old_file}")
                        except Exception as e:
                            logger.error(f"處理檔案 {old_file} 時發生錯誤: {str(e)}")
                else:
                    logger.info("取消刪除操作")
        else:
            logger.info("沒有找到需要刪除的檔案")
        
        logger.info("檔案刪除測試完成")
        
    except Exception as e:
        logger.error(f"測試檔案刪除功能時發生錯誤: {str(e)}")

def main():
    """主程序"""
    logger = None
    try:
        # 初始化配置
        config = FeatureConfig()
        
        # 設定日誌
        logger = setup_logging(config.log_path)
        
        # 驗證配置
        if not validate_config(config, logger):
            logger.error("配置驗證失敗，程序終止")
            sys.exit(1)
        
        # 添加配置文件存在性檢查
        if not config.project_path.exists():
            raise FileNotFoundError(f"找不到專案目錄：{config.project_path}")
            
        # 添加數據目錄檢查
        if not config.data_path.exists():
            raise FileNotFoundError(f"找不到數據目錄：{config.data_path}")
        
        # 記錄開始時間
        start_time = datetime.now()
        logger.info("開始特徵生成流程...")
        logger.info(f"處理時間範圍: {config.TEST_SETTING['start_date']} 到 {config.TEST_SETTING['end_date']}")
        logger.info(f"處理股票數量: {len(config.TEST_SETTING['test_stocks'])}")
        
        # 初始化特徵管理器
        logger.info("初始化特徵管理器...")
        manager = FeatureManager(config)
        
        # 生成特徵
        logger.info("開始生成特徵...")
        if not manager.generate_features():
            logger.error("特徵生成失敗")
            return
            
        logger.info("特徵生成完成")
        
        # 驗證特徵
        logger.info("開始驗證特徵...")
        validator = FeatureValidator(config)
        if not validator.validate_features():
            logger.warning("特徵驗證發現問題")
            return
            
        logger.info("特徵驗證通過")
        
        # 檢查特徵檔案
        logger.info("開始檢查特徵檔案...")
        check_features(config, logger, config.TEST_SETTING['test_stocks'])
        
        # 記錄結束時間和執行時間
        end_time = datetime.now()
        execution_time = end_time - start_time
        logger.info(f"特徵生成流程結束，總執行時間: {execution_time}")
        
        # 記錄記憶體使用情況
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        logger.info(f"最終記憶體使用: {memory_info.rss / 1024 / 1024:.2f} MB")
        
    except ImportError as e:
        if logger:
            logger.error(f"模組導入錯誤: {str(e)}")
        else:
            print(f"錯誤: 模組導入失敗: {str(e)}")
    except FileNotFoundError as e:
        if logger:
            logger.error(f"找不到必要的文件: {str(e)}")
        else:
            print(f"錯誤: 找不到必要的文件: {str(e)}")
    except PermissionError as e:
        if logger:
            logger.error(f"權限錯誤: {str(e)}")
        else:
            print(f"錯誤: 權限不足: {str(e)}")
    except Exception as e:
        if logger:
            logger.error(f"執行過程中發生錯誤: {str(e)}", exc_info=True)
        else:
            print(f"錯誤: {str(e)}")
        raise
    finally:
        if logger:
            logger.info("特徵生成流程結束")

if __name__ == "__main__":
    main() 