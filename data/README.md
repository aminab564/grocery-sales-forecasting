# Data Download Instructions

This project uses the **Store Sales - Time Series Forecasting** dataset from Kaggle.

## Required Files

### 1. Download from Kaggle

**URL**: https://www.kaggle.com/c/store-sales-time-series-forecasting/data

You will need a Kaggle account. If you don't have one, create a free account at https://www.kaggle.com

### 2. Required CSV Files

Download and place these files in the `data/favorita/` directory:

- ✅ `train.csv` (~500 MB) - Training data with sales
- ✅ `test.csv` (~50 MB) - Test data for competition
- ✅ `oil.csv` (~5 KB) - Daily oil prices
- ✅ `holidays_events.csv` (~10 KB) - Ecuador holidays and events
- ✅ `stores.csv` (~2 KB) - Store metadata
- ⚠️ `transactions.csv` (optional) - Daily transactions per store

### 3. Download Options

**Option A: Web Interface**
1. Visit the competition data page
2. Click "Download All" button
3. Extract the ZIP file
4. Copy CSV files to `data/favorita/`

**Option B: Kaggle API** (Recommended)
```bash
# Install Kaggle API
pip install kaggle

# Setup API credentials (follow Kaggle docs)
# Download dataset
kaggle competitions download -c store-sales-time-series-forecasting

# Extract
unzip store-sales-time-series-forecasting.zip -d data/favorita/
```

## Directory Structure After Download

```
data/
├── README.md (this file)
└── favorita/
    ├── train.csv              # 3.3M rows
    ├── test.csv               # 28K rows
    ├── oil.csv                # 1,218 rows
    ├── holidays_events.csv    # 350 rows
    ├── stores.csv             # 54 rows
    └── transactions.csv       # (optional)
```

## Processed Data

The notebooks will generate processed files in `data/processed/`:

- `train_subset.csv` - 150 time series subset (generated in Part 1)
- `train_with_features.csv` - Data with engineered features (generated in Part 2)
- `subset_metadata.json` - Configuration file
- `feature_config.json` - Feature engineering metadata

**Note**: Processed files are NOT in the git repository. They will be created when you run the notebooks.

## Verification

After downloading, verify you have the correct files:

```python
import pandas as pd
from pathlib import Path

data_path = Path('data/favorita')

files = {
    'train.csv': 3000000,  # Approximate row count
    'test.csv': 28000,
    'oil.csv': 1200,
    'holidays_events.csv': 300,
    'stores.csv': 50
}

for filename, expected_rows in files.items():
    filepath = data_path / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"✅ {filename}: {len(df):,} rows")
    else:
        print(f"❌ {filename}: NOT FOUND")
```

Expected output:
```
✅ train.csv: 3,000,888 rows
✅ test.csv: 28,512 rows
✅ oil.csv: 1,218 rows
✅ holidays_events.csv: 350 rows
✅ stores.csv: 54 rows
```

## Dataset Information

### Train Data Schema
- `id`: Row identifier
- `date`: Date of sale
- `store_nbr`: Store number (1-54)
- `family`: Product family/category
- `sales`: Units sold (target variable)
- `onpromotion`: Number of items on promotion

### Additional Information
- **Time Period**: 2013-01-01 to 2017-08-15
- **Stores**: 54 locations across Ecuador
- **Product Families**: 33 categories
- **Total Observations**: 3M+ daily store-product records

## Troubleshooting

### Issue: "Kaggle API not authenticated"
**Solution**: 
1. Go to Kaggle Account Settings
2. Create new API token
3. Download `kaggle.json`
4. Place in `~/.kaggle/` directory

### Issue: "File size too large"
**Solution**: 
- Use stable internet connection
- Download files individually instead of "Download All"
- Ensure sufficient disk space (~600 MB)

### Issue: "Cannot access competition data"
**Solution**:
- Accept competition rules on Kaggle website
- Ensure you're logged in
- Check if competition is still active

## Data Usage Agreement

By downloading this data, you agree to:
- Use it for educational/research purposes only
- Follow Kaggle's Terms of Service
- Not redistribute the raw data files

## Questions?

If you encounter issues downloading the data:
1. Check Kaggle's [API documentation](https://github.com/Kaggle/kaggle-api)
2. Visit the [competition discussion forum](https://www.kaggle.com/c/store-sales-time-series-forecasting/discussion)
3. Open an issue in this repository

---

**Last Updated**: November 2025
