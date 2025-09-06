import pandas as pd
from ydata_profiling import ProfileReport

def generate_report(csv_file, output_html):
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Loaded: {csv_file} | Shape: {df.shape}")
        if df.empty:
            print(f"❌ {csv_file} is empty.")
            return
        profile = ProfileReport(df, title=f"EDA Report for {csv_file}", explorative=True)
        profile.to_file(f"templates/{output_html}")
        print(f"✅ EDA report saved to templates/{output_html}")
    except Exception as e:
        print(f"❌ Failed to generate EDA for {csv_file}: {e}")

# Generate separate reports
generate_report("bigmart_data.csv", "eda_bigmart.html")
generate_report("rossmann_train.csv", "eda_rossmann_train.html")
generate_report("rossmann_store.csv", "eda_rossmann_store.html")
