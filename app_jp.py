import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- データ読み込み関数 ---
def load_all_series():
    def load_single_series(file_path, col_name):
        xls = pd.ExcelFile(file_path)
        df_all = []
        for sheet in reversed(xls.sheet_names):
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, usecols=[0, 1])
                df.columns = ["Date", col_name]
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                df = df.dropna(subset=["Date", col_name])
                df = df.set_index("Date").sort_index()
                if not df.empty:
                    df_all.append(df)
            except Exception:
                continue
        df_all = pd.concat(df_all).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="first")]
        return df_all

    nik = load_single_series("data/Nikkei.xlsx", "Nikkei")
    dow = load_single_series("data/Dow.xlsx", "Dow")
    jgb = load_single_series("data/JGB10Y.xlsx", "JGB10Y")
    kospi = load_single_series("data/KOSPI.xlsx", "KOSPI")

    df = pd.concat([nik, dow, jgb, kospi], axis=1).sort_index()
    df = df.dropna()
    return df

# --- 特徴量作成 ---
def make_features(df):
    df["RET_NIK"] = df["Nikkei"].pct_change()
    df["RET_NIK_L1"] = df["RET_NIK"].shift(1)
    df["RET_NIK_L2"] = df["RET_NIK"].shift(2)
    df["RET_NIK_MA3"] = df["RET_NIK"].rolling(3).mean()
    df["RET_NIK_STD3"] = df["RET_NIK"].rolling(3).std()

    df["RET_DOW"] = df["Dow"].pct_change()
    df["RET_DOW_L1"] = df["RET_DOW"].shift(1)
    df["RET_DOW_L2"] = df["RET_DOW"].shift(2)
    df["RET_DOW_MA3"] = df["RET_DOW"].rolling(3).mean()
    df["RET_DOW_STD3"] = df["RET_DOW"].rolling(3).std()

    df["DY_JGB"] = df["JGB10Y"].diff()
    df["DY_JGB_L1"] = df["DY_JGB"].shift(1)
    df["DY_JGB_L2"] = df["DY_JGB"].shift(2)
    df["DY_JGB_MA3"] = df["DY_JGB"].rolling(3).mean()
    df["DY_JGB_STD3"] = df["DY_JGB"].rolling(3).std()

    df["RET_KOSPI"] = df["KOSPI"].pct_change()
    df["RET_KOSPI_L1"] = df["RET_KOSPI"].shift(1)
    df["RET_KOSPI_L2"] = df["RET_KOSPI"].shift(2)
    df["RET_KOSPI_MA3"] = df["RET_KOSPI"].rolling(3).mean()
    df["RET_KOSPI_STD3"] = df["RET_KOSPI"].rolling(3).std()

    df["RET_NIK_NEXT"] = df["RET_NIK"].shift(-1)
    df_feat = df.dropna()

    feature_cols = [
        "RET_NIK_L1", "RET_NIK_L2", "RET_NIK_MA3", "RET_NIK_STD3",
        "RET_DOW_L1", "RET_DOW_L2", "RET_DOW_MA3", "RET_DOW_STD3",
        "DY_JGB_L1", "DY_JGB_L2", "DY_JGB_MA3", "DY_JGB_STD3",
        "RET_KOSPI_L1", "RET_KOSPI_L2", "RET_KOSPI_MA3", "RET_KOSPI_STD3"
    ]
    return df_feat, feature_cols

# --- 月予測関数 ---
def summarize_month(df_feat, feature_cols, month_start, month_end):
    X_all = df_feat[feature_cols]
    y_all = df_feat["RET_NIK_NEXT"]
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X_all)
    model = ElasticNet(alpha=0.001, l1_ratio=0.1, random_state=42)
    model.fit(X_scaled_all, y_all)

    results = []
    dates = []
    for date in pd.date_range(start=month_start, end=month_end):
        if date in df_feat.index:
            X_target = df_feat.loc[[date], feature_cols]
            X_target_scaled = scaler.transform(X_target)
            pred = model.predict(X_target_scaled)[0]
            results.append(pred)
            dates.append(date)

    if not results:
        return None, "⚠️ この月の予測に必要なデータが不足しています。"

    avg_pred = np.mean(results)
    direction = "📈　上昇傾向" if avg_pred > 0 else "📉 下落傾向"

    summary = f"""
    ## Monthly Forecast Summary
    - 平均予測値: `{avg_pred:.5f}`
    - 傾向: {direction}
    """
    return pd.DataFrame({"日付": dates, "予測値": results}).set_index("Date"), summary

# --- Streamlit UI ---
st.set_page_config(page_title="月ごとの予測アプリ", page_icon="📅")
st.title("📅月ごとの予測アプリ")

df_all = load_all_series()
df_all = df_all.apply(pd.to_numeric, errors="coerce")
df_all = df_all.dropna()

if df_all.empty:
    st.error("❌ データが読み込めませんでした。Excelファイルを確認してください。")
    st.stop()

df_feat, feature_cols = make_features(df_all)

if df_feat.empty:
    st.error("❌ 特徴量が生成できませんでした。元データに問題がある可能性があります。")
    st.stop()

today = datetime.today()
available_months = sorted(set(df_feat.index.strftime("%Y-%m")))
selected_month = st.selectbox("予測したい月を選択", available_months)

month_start = pd.to_datetime(f"{selected_month}-01")
month_end = month_start + pd.offsets.MonthEnd(0)

if st.button("予測を実行"):
    with st.spinner("🔄 計算中です。もうしばらくお待ちください。"):
        df_pred, summary = summarize_month(df_feat, feature_cols, month_start, month_end)
        st.markdown(summary)
        if df_pred is not None:
            st.line_chart(df_pred["Prediction"])

latest_date = df_feat.index.max()
st.caption(f"📌 最新のデータ: {latest_date.date()}")