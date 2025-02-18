import csv
import os

# 定義目標路徑
output_path = r"D:\Min\Python\Project\FA_Data\meta_data\data_sources.csv"

# 確保目錄存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 定義數據源信息
data_sources = [
    {
        "資料類型": "股票交易資料",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/meta_data/all_stocks_data.csv",
        "更新頻率": "每日",
        "資料格式": "CSV",
        "欄位說明": "證券代號(str),證券名稱(str),成交股數(int),成交筆數(int),成交金額(int),開盤價(float),最高價(float),最低價(float),收盤價(float),漲跌(+/-)(str),漲跌價差(float),最後揭示買價(float),最後揭示買量(int),最後揭示賣價(float),最後揭示賣量(int),本益比(float),日期(str:YYYY-MM-DD),SMA30(float),DEMA30(float),EMA30(float),RSI(float),MACD(float),MACD_signal(float),MACD_hist(float),slowk(float),slowd(float),TSF(float),middleband(float),SAR(float)"
    },
    {
        "資料類型": "公司基本資料",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/meta_data/companies.csv",
        "更新頻率": "不定期",
        "資料格式": "CSV",
        "欄位說明": "industry_category(str),stock_id(str),stock_name(str),type(str),date(str:YYYY-MM-DD),download_time(str:YYYY-MM-DD HH:mm:ss)"
    },
    {
        "資料類型": "產業指數資料",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/meta_data/industry_index.csv",
        "更新頻率": "每日",
        "資料格式": "CSV",
        "欄位說明": "指數名稱(str),收盤指數(float),漲跌(str),漲跌點數(float),漲跌百分比(float),日期(str:YYYY-MM-DD)"
    },
    {
        "資料類型": "台灣加權指數",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/meta_data/market_index.csv",
        "更新頻率": "每日",
        "資料格式": "CSV",
        "欄位說明": "Date(str:YYYY-MM-DD),Open(float),High(float),Low(float),Close(float),Adj Close(float),Volume(int)"
    },
    {
        "資料類型": "週度產業相關性分析",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/industry_correlation/weekly/industry_correlation_{date}.csv",
        "更新頻率": "週",
        "資料格式": "CSV",
        "欄位說明": "industry_1(str),industry_2(str),correlation(float),p_value(float),sample_size(int)"
    },
    {
        "資料類型": "月度產業相關性分析",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/industry_correlation/monthly/industry_correlation_{date}.csv",
        "更新頻率": "月",
        "資料格式": "CSV",
        "欄位說明": "industry_1(str),industry_2(str),correlation(float),p_value(float),sample_size(int)"
    },
    {
        "資料類型": "產業價格指數分析",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/industry_analysis/price_index/{industry}_{start_date}_{end_date}_{generate_date}.json",
        "更新頻率": "每日",
        "資料格式": "JSON",
        "欄位說明": "correlation_matrix(dict),technical_analysis(dict),investment_suggestions(dict),metadata(dict)"
    },
    {
        "資料類型": "產業報酬指數分析",
        "檔案路徑": r"D:/Min/Python/Project/FA_Data/industry_analysis/return_index/{industry}_{start_date}_{end_date}_{generate_date}.json",
        "更新頻率": "每日",
        "資料格式": "JSON",
        "欄位說明": "correlation_matrix(dict),technical_analysis(dict),investment_suggestions(dict),metadata(dict)"
    }
]

# 寫入 CSV 文件
with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=["資料類型", "檔案路徑", "更新頻率", "資料格式", "欄位說明"])
    writer.writeheader()
    writer.writerows(data_sources)

print(f"數據源記錄文件已創建：{output_path}") 