import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# --- データ読み込み関数 ---
def 読み込み_全シリーズ():
    def 読み込み_単シリーズ(file_path, col_name):
        xls = pd.ExcelFile(file_path)
        df_all = []
        for sheet in reversed(xls.sheet_names):
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, usecols=[0, 1])
                df.columns = ["日付", col_name]
                df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
                df = df.dropna(subset=["日付", col_name])
                df = df.set_index("日付").sort_index()
                if not df.empty:
                    df_all.append(df)
            except Exception:
                continue
        df_all = pd.concat(df_all).sort_index()
        df_all = df_all[~df_all.index.duplicated(keep="first")]
        return df_all

    nik = 読み込み_単シリーズ("data/Nikkei.xlsx", "日経平均")
    dow = 読み込み_単シリーズ("data/Dow.xlsx", "ダウ平均")
    jgb = 読み込み_単シリーズ("data/JGB10Y.xlsx", "長期金利")
    kospi = 読み込み_単シリーズ("data/KOSPI.xlsx", "KOSPI")

    df = pd.concat([nik, dow, jgb, kospi], axis=1).sort_index()
    df = df.dropna()
    return df

# --- 特徴量作成 ---
def 特徴量作成(df):
    df["日経リターン"] = df["日経平均"].pct_change()
    df["日経_L1"] = df["日経リターン"].shift(1)
    df["日経_L2"] = df["日経リターン"].shift(2)
    df["日経_MA3"] = df["日経リターン"].rolling(3).mean()
    df["日経_STD3"] = df["日経リターン"].rolling(3).std()

    df["ダウリターン"] = df["ダウ平均"].pct_change()
    df["ダウ_L1"] = df["ダウリターン"].shift(1)
    df["ダウ_L2"] = df["ダウリターン"].shift(2)
    df["ダウ_MA3"] = df["ダウリターン"].rolling(3).mean()
    df["ダウ_STD3"] = df["ダウリターン"].rolling(3).std()

    df["金利変化"] = df["長期金利"].diff()
    df["金利_L1"] = df["金利変化"].shift(1)
    df["金利_L2"] = df["金利変化"].shift(2)
    df["金利_MA3"] = df["金利変化"].rolling(3).mean()
    df["金利_STD3"] = df["金利変化"].rolling(3).std()

    df["KOSPIリターン"] = df["KOSPI"].pct_change()
    df["KOSPI_L1"] = df["KOSPIリターン"].shift(1)
    df["KOSPI_L2"] = df["KOSPIリターン"].shift(2)
    df["KOSPI_MA3"] = df["KOSPIリターン"].rolling(3).mean()
    df["KOSPI_STD3"] = df["KOSPIリターン"].rolling(3).std()

    df["翌日経リターン"] = df["日経リターン"].shift(-1)
    df_feat = df.dropna()

    feature_cols = [
        "日経_L1", "日経_L2", "日経_MA3", "日経_STD3",
        "ダウ_L1", "ダウ_L2", "ダウ_MA3", "ダウ_STD3",
        "金利_L1", "金利_L2", "金利_MA3", "金利_STD3",
        "KOSPI_L1", "KOSPI_L2", "KOSPI_MA3", "KOSPI_STD3"
    ]
    return df_feat, feature_cols

# --- 月間予測 ---
def 月間予測(df_feat, feature_cols, month_start, month_end):
    X_all = df_feat[feature_cols]
    y_all = df_feat["翌日経リターン"]

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
    direction = "📈 上昇傾向" if avg_pred > 0 else "📉 下落傾向"

    summary = f"""
    ## 📅 月間予測
    - 平均予測値: `{avg_pred:.5f}`
    - 傾向: {direction}
    """

    return pd.DataFrame({"日付": dates, "予測値": results}).set_index("日付"), summary

# --- Streamlit UI ---
st.set_page_config(page_title="月ごとの予測アプリ", page_icon="📅")
st.title("📅 月ごとの予測アプリ")

df_all = 読み込み_全シリーズ()
df_all = df_all.apply(pd.to_numeric, errors="coerce").dropna()

if df_all.empty:
    st.error("❌ データが読み込めませんでした。Excelファイルを確認してください。")
    st.stop()

df_feat, feature_cols = 特徴量作成(df_all)

if df_feat.empty:
    st.error("❌ 特徴量が生成できませんでした。元データに問題がある可能性があります。")
    st.stop()

available_months = sorted(set(df_feat.index.strftime("%Y-%m")))
selected_month = st.selectbox("予測したい月を選択してください", available_months)

month_start = pd.to_datetime(f"{selected_month}-01")
month_end = month_start + pd.offsets.MonthEnd(0)

if st.button("予測を実行"):
    with st.spinner("🔄 計算中です。しばらくお待ちください…"):
        df_pred, summary = 月間予測(df_feat, feature_cols, month_start, month_end)
        st.markdown(summary)
        if df_pred is not None:
            st.line_chart(df_pred["予測値"])

latest_date = df_feat.index.max()
st.caption(f"📌 使用しているデータの最終日: {latest_date.date()}")