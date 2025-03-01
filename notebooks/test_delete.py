import os
from pathlib import Path
import time
import csv

def test_delete_files(auto_confirm=True):
    """測試刪除檔案功能
    
    Args:
        auto_confirm: 是否自動確認刪除檔案
    """
    # 設定特徵檔案路徑
    features_dir = Path("D:/Min/Python/Project/FA_Data/features")
    
    # 測試股票代碼和產業
    stock_id = "2330"
    industry = "半導體業"
    diff_industry = "電子工業"
    
    # 測試日期範圍
    start_date = "20230103"
    end_date = "20241112"
    diff_date_range_start = "20240101"
    diff_date_range_end = "20241230"
    
    # 產生日期
    old_date = "20250101"  # 舊的產生日期
    new_date = "20250228"  # 新的產生日期
    
    # 檔案名稱
    old_filename = f"{stock_id}_{industry}_{start_date}_{end_date}_{old_date}.csv"
    diff_industry_filename = f"{stock_id}_{diff_industry}_{start_date}_{end_date}_{old_date}.csv"
    diff_date_filename = f"{stock_id}_{industry}_{diff_date_range_start}_{diff_date_range_end}_{old_date}.csv"
    new_filename = f"{stock_id}_{industry}_{start_date}_{end_date}_{new_date}.csv"
    
    # 檔案路徑
    old_filepath = features_dir / old_filename
    diff_industry_filepath = features_dir / diff_industry_filename
    diff_date_filepath = features_dir / diff_date_filename
    new_filepath = features_dir / new_filename
    
    # 刪除現有的測試檔案
    for filepath in [old_filepath, diff_industry_filepath, diff_date_filepath, new_filepath]:
        if filepath.exists():
            try:
                os.chmod(str(filepath), 0o777)
                filepath.unlink()
                print(f"已刪除現有的檔案: {filepath}")
            except Exception as e:
                print(f"刪除現有檔案時發生錯誤: {str(e)}")
    
    # 創建測試檔案
    def create_test_file(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'stock_id', 'stock_name', 'open', 'high', 'low', 'close', 'volume'])
            writer.writerow(['2023-01-03', stock_id, 'TSMC', 100, 105, 98, 102, 1000000])
            writer.writerow(['2023-01-04', stock_id, 'TSMC', 102, 107, 100, 105, 1200000])
            writer.writerow(['2023-01-05', stock_id, 'TSMC', 105, 110, 103, 108, 1500000])
    
    # 創建舊檔案（相同股票、相同產業、相同日期範圍）
    create_test_file(old_filepath)
    print(f"已創建測試檔案: {old_filepath}")
    
    # 創建不同產業的測試檔案
    create_test_file(diff_industry_filepath)
    print(f"已創建不同產業的測試檔案: {diff_industry_filepath}")
    
    # 創建不同日期範圍的測試檔案
    create_test_file(diff_date_filepath)
    print(f"已創建不同日期範圍的測試檔案: {diff_date_filepath}")
    
    # 創建新檔案
    create_test_file(new_filepath)
    print(f"已創建新檔案: {new_filepath}")
    
    # 等待一小段時間，確保檔案系統操作完成
    time.sleep(1)
    
    # 模擬feature_manager.py中的save_features方法
    print("\n開始測試刪除功能...")
    
    # 使用更精確的模式匹配，匹配股票代碼、產業和日期範圍
    pattern = f"{stock_id}_{industry}_{start_date}_{end_date}_*.csv"
    print(f"搜尋檔案模式: {pattern}")
    existing_files = list(features_dir.glob(pattern))
    print(f"找到 {len(existing_files)} 個檔案:")
    
    # 顯示找到的檔案
    for i, file in enumerate(existing_files):
        print(f"{i+1}. {file.name}")
    
    # 過濾出需要刪除的檔案（排除當前檔案）
    files_to_delete = []
    current_file = new_filepath  # 當前檔案是新創建的檔案
    
    for file in existing_files:
        if file != current_file:
            print(f"找到需要刪除的舊檔案: {file.name}")
            files_to_delete.append(file)
    
    print(f"\n找到 {len(files_to_delete)} 個需要刪除的舊檔案:")
    for i, file in enumerate(files_to_delete):
        print(f"{i+1}. {file.name}")
    
    # 詢問是否刪除檔案
    if files_to_delete:
        if auto_confirm:
            choice = 'y'
            print("\n自動確認刪除檔案")
        else:
            choice = input("\n是否刪除這些檔案？(y/n): ")
        
        if choice.lower() == 'y':
            for old_file in files_to_delete:
                try:
                    print(f"嘗試刪除檔案: {old_file}")
                    if old_file.exists():
                        try:
                            os.chmod(str(old_file), 0o777)  # 嘗試更改權限
                            print(f"已更改檔案權限: {old_file}")
                        except Exception as e:
                            print(f"更改檔案權限時發生錯誤: {str(e)}")
                        
                        deleted = False
                        
                        # 方法1: 使用pathlib的unlink
                        try:
                            old_file.unlink()
                            print(f"成功使用unlink刪除檔案: {old_file}")
                            deleted = True
                        except Exception as e:
                            print(f"使用unlink刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                        
                        # 方法2: 使用os.remove
                        if not deleted:
                            try:
                                os.remove(str(old_file))
                                print(f"使用os.remove成功刪除檔案: {old_file}")
                                deleted = True
                            except Exception as e:
                                print(f"使用os.remove刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                        
                        # 方法3: 使用os.unlink
                        if not deleted:
                            try:
                                os.unlink(str(old_file))
                                print(f"使用os.unlink成功刪除檔案: {old_file}")
                                deleted = True
                            except Exception as e:
                                print(f"使用os.unlink刪除檔案 {old_file} 時發生錯誤: {str(e)}")
                        
                        # 檢查是否成功刪除
                        if not deleted:
                            print(f"警告: 無法刪除檔案: {old_file}")
                    else:
                        print(f"檔案不存在: {old_file}")
                except Exception as e:
                    print(f"處理檔案 {old_file} 時發生錯誤: {str(e)}")
        else:
            print("取消刪除操作")
    else:
        print("沒有找到需要刪除的檔案")
    
    # 檢查結果
    print("\n檢查刪除結果...")
    
    # 檢查相同產業、相同日期範圍的檔案
    same_pattern = f"{stock_id}_{industry}_{start_date}_{end_date}_*.csv"
    same_files = list(features_dir.glob(same_pattern))
    print(f"相同產業、相同日期範圍的檔案: {len(same_files)} 個")
    for i, file in enumerate(same_files):
        print(f"{i+1}. {file.name}")
    
    # 檢查不同產業的檔案
    diff_industry_pattern = f"{stock_id}_{diff_industry}_{start_date}_{end_date}_*.csv"
    diff_industry_files = list(features_dir.glob(diff_industry_pattern))
    print(f"\n不同產業的檔案: {len(diff_industry_files)} 個")
    for i, file in enumerate(diff_industry_files):
        print(f"{i+1}. {file.name}")
    
    # 檢查不同日期範圍的檔案
    diff_date_pattern = f"{stock_id}_{industry}_{diff_date_range_start}_{diff_date_range_end}_*.csv"
    diff_date_files = list(features_dir.glob(diff_date_pattern))
    print(f"\n不同日期範圍的檔案: {len(diff_date_files)} 個")
    for i, file in enumerate(diff_date_files):
        print(f"{i+1}. {file.name}")
    
    # 驗證結果
    print("\n驗證結果:")
    if len(same_files) == 1 and same_files[0].name == new_filename:
        print("✓ 成功: 只保留了最新的相同產業、相同日期範圍的檔案")
    else:
        print("✗ 失敗: 相同產業、相同日期範圍的檔案刪除有問題")
    
    if len(diff_industry_files) == 1:
        print("✓ 成功: 保留了不同產業的檔案")
    else:
        print("✗ 失敗: 不同產業的檔案被錯誤刪除")
    
    if len(diff_date_files) == 1:
        print("✓ 成功: 保留了不同日期範圍的檔案")
    else:
        print("✗ 失敗: 不同日期範圍的檔案被錯誤刪除")

if __name__ == "__main__":
    test_delete_files(auto_confirm=True) 