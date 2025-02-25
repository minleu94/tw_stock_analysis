# 台灣股票產業分析與特徵生成系統

這個系統用於分析台灣股市的產業指數，生成相關特徵，並進行驗證。

## 系統功能

1. **產業分析**：分析台灣股市各產業的表現，包括趨勢分析、風險分析和輪動分析。
2. **特徵生成**：基於產業分析結果，生成用於後續模型訓練的特徵。
3. **特徵驗證**：驗證生成的特徵是否符合要求，包括文件結構、內容和數據完整性。

## 系統架構

系統主要由以下幾個模塊組成：

- **FeatureConfig**：配置管理，包括路徑設置、參數設置等。
- **IndustryAnalysisSystem**：產業分析系統，負責分析產業指數的表現。
- **FeatureManager**：特徵管理系統，負責生成特徵。
- **FeatureValidator**：特徵驗證系統，負責驗證特徵的有效性。

## 目錄結構

```
tw_stock_analysis/
├── feature_config.py       # 配置管理
├── feature_manager.py      # 特徵管理
├── feature_validator.py    # 特徵驗證
├── industry_analysis.py    # 產業分析
├── run_feature_pipeline.py # 特徵生成與驗證流程
├── validate_features.py    # 特徵驗證入口
├── logs/                   # 日誌目錄
└── notebooks/              # Jupyter筆記本
```

## 使用方法

### 1. 產業分析

執行產業分析，生成產業分析結果：

```bash
python industry_analysis.py
```

### 2. 特徵生成

基於產業分析結果，生成特徵：

```bash
python feature_manager.py
```

### 3. 特徵驗證

驗證生成的特徵：

```bash
python validate_features.py
```

### 4. 完整流程

執行完整的特徵生成與驗證流程：

```bash
python run_feature_pipeline.py
```

## 配置說明

系統的配置在 `feature_config.py` 中定義，主要包括：

- **路徑設置**：數據路徑、特徵路徑、日誌路徑等。
- **參數設置**：技術指標參數、測試設置等。
- **文件格式**：特徵文件格式、產業分析文件格式等。

## 數據要求

系統需要以下數據：

1. **產業指數數據**：包括價格指數和報酬指數。
2. **公司資料**：包括股票代碼、公司名稱、產業分類等。
3. **產業對照檔案**：產業名稱與產業指數的對照關係。

## 輸出結果

系統的輸出結果包括：

1. **產業分析結果**：JSON格式，包含產業的趨勢分析、風險分析和輪動分析。
2. **特徵文件**：CSV格式，包含基本價格特徵、技術指標和產業特徵。
3. **驗證報告**：TXT格式，包含特徵驗證的結果。

## 注意事項

1. 請確保數據目錄中包含必要的基礎數據文件。
2. 系統使用 UTF-8-SIG 編碼，請確保數據文件的編碼一致。
3. 日誌文件會保存在 logs 目錄中，可用於排查問題。

## 依賴套件

- pandas
- numpy
- talib
- scikit-learn
- scipy
- tqdm
- psutil

## 安裝依賴

```bash
pip install pandas numpy scikit-learn scipy tqdm psutil
pip install ta-lib  # 可能需要特殊安裝步驟，請參考官方文檔
``` 