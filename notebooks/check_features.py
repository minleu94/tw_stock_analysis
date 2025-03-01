import pandas as pd
import os
from pathlib import Path
import sys

def main():
    # 設定特徵檔案路徑
    features_dir = Path("D:/Min/Python/Project/FA_Data/features")

    # 查找最新的2330和2317特徵檔案
    files = list(features_dir.glob("*.csv"))
    files = [f for f in files if ("2330" in f.name or "2317" in f.name)]
    files = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

    if not files:
        print("找不到2330或2317的特徵檔案")
        exit(1)

    # 設定顯示選項
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.float_format', '{:.4f}'.format)

    # 讀取最新的特徵檔案
    for file in files[:4]:  # 只讀取最新的四個檔案
        print(f"\n{'='*80}")
        print(f"處理檔案: {file.name}")
        print(f"{'='*80}")
        try:
            df = pd.read_csv(file)
            
            # 顯示基本資訊
            print(f"資料形狀: {df.shape}")
            
            # 檢查日期欄位名稱
            date_col = None
            for col in ['date', '日期', 'Date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                print(f"資料日期範圍: {df[date_col].min()} 到 {df[date_col].max()}")
            else:
                print("找不到日期欄位")
            
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
                print(f"股票代碼: {df[stock_id_col].iloc[0]}")
            else:
                print("找不到股票代碼欄位")
            
            if stock_name_col:
                print(f"股票名稱: {df[stock_name_col].iloc[0]}")
            else:
                print("找不到股票名稱欄位")
            
            # 顯示所有欄位
            all_columns = df.columns.tolist()
            print(f"\n總共有 {len(all_columns)} 個欄位:")
            for i in range(0, len(all_columns), 5):
                print(', '.join(all_columns[i:i+5]))
            
            # 顯示前3行基本數據
            print("\n前3行基本數據:")
            basic_cols = []
            if date_col:
                basic_cols.append(date_col)
            if stock_id_col:
                basic_cols.append(stock_id_col)
            if stock_name_col:
                basic_cols.append(stock_name_col)
            
            # 檢查價格和成交量欄位
            price_cols = []
            for col_name in ['open', 'high', 'low', 'close', 'volume', '開盤價', '最高價', '最低價', '收盤價', '成交量']:
                if col_name in df.columns:
                    price_cols.append(col_name)
                    basic_cols.append(col_name)
            
            if basic_cols:
                print(df.head(3)[basic_cols])
            else:
                print("找不到基本數據欄位")
            
            # 顯示技術指標欄位
            tech_cols = ['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'slowk', 'slowd', 'SAR', 'TSF', 'middleband', 'SMA30', 'EMA30', 'DEMA30']
            available_tech_cols = [col for col in tech_cols if col in df.columns]
            if available_tech_cols:
                print("\n前3行技術指標數據:")
                print(df.head(3)[available_tech_cols])
            
            # 顯示產業相關欄位
            industry_cols = [col for col in df.columns if '產業_' in col or 'industry_' in col]
            if industry_cols:
                print("\n產業相關欄位:")
                for col in industry_cols:
                    print(f"- {col}")
                
                print("\n前1行產業特徵數據:")
                industry_data = df.head(1)[industry_cols].T  # 轉置以便更好地顯示
                industry_data.columns = ['值']
                print(industry_data)
            
            # 顯示一些統計信息
            close_col = None
            for col in ['close', '收盤價', 'Close', 'CLOSE']:
                if col in df.columns:
                    close_col = col
                    break
            
            if close_col:
                print(f"\n收盤價統計:")
                print(df[close_col].describe())
                
                # 顯示最近10天的數據
                print("\n最近10天的收盤價和技術指標:")
                cols_to_show = []
                if date_col:
                    cols_to_show.append(date_col)
                cols_to_show.append(close_col)
                
                for tech_col in ['RSI', 'MACD']:
                    if tech_col in df.columns:
                        cols_to_show.append(tech_col)
                
                recent_data = df.tail(10)[cols_to_show]
                print(recent_data)
            
        except Exception as e:
            print(f"讀取檔案 {file.name} 時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n特徵檔案分析完成")

if __name__ == "__main__":
    main() 