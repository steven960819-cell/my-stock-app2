import streamlit as st
import pandas as pd
import requests

# ==============================================================================
# 1. 全新自動化股票與 ETF 爬蟲整合 (取代原本冗長的手打清單)
# ==============================================================================
@st.cache_data(ttl=86400)  # 快取 24 小時，避免頻繁請求導致網頁卡頓
def fetch_taiwan_stocks():
    urls = {
        "上市": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "上櫃": "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    }
    
    stock_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for market_type, url in urls.items():
        try:
            # 抓取網頁並修正為 CP950/MS950 編碼
            response = requests.get(url, headers=headers)
            response.encoding = 'ms950'
            
            # 使用 pandas 解析 HTML 中的表格
            dfs = pd.read_html(response.text)
            df = dfs[0]
            
            # 將第一行設定為標題，並去除舊的第一行
            df.columns = df.iloc[0]
            df = df.iloc[1:]
            
            # 確保有價證券欄位有資料
            df = df.dropna(subset=['有價證券代號及名稱'])
            
            is_valid_section = False  # 區段標記開關
            
            for _, row in df.iterrows():
                text = str(row['有價證券代號及名稱']).strip()
                
                # 允許「股票」與「受益憑證 (ETF)」區段進入抓取
                if "股 票" in text or "股票" in text or "受益憑證" in text:
                    is_valid_section = True
                    continue
                # 如果遇到權證、債券等雜訊大標題，立即關閉開關
                elif "認購" in text or "認售" in text or "權證" in text or "債券" in text or "創櫃" in text:
                    is_valid_section = False
                    continue
                
                # 解析符合條件的行
                if is_valid_section and "　" in text:
                    parts = text.split("　")
                    stock_id = parts[0].strip()
                    stock_name = parts[1].strip()
                    
                    # 精準過濾防線：完美保留個股 + ETF，封殺權證
                    if stock_id.isdigit():
                        # 條件一：剛好 4 碼（一般股票）
                        is_normal_stock = (len(stock_id) == 4)
                        # 條件二：5 或 6 碼，且必須是 00 開頭（精準保留 ETF）
                        is_etf = (len(stock_id) in [5, 6] and stock_id.startswith("00"))
                        
                        if is_normal_stock or is_etf:
                            # 排除特別股與非普通股性質的雜訊
                            if "特" not in stock_name and "債" not in stock_name:
                                stock_list.append({"id": stock_id, "name": stock_name})
                            
        except Exception as e:
            st.error(f"無法從 {market_type} 網址獲取資料: {e}")
            
    # 依據代碼由小到大排序
    stock_list = sorted(stock_list, key=lambda x: x['id'])
    return stock_list

# 啟動自動下載與篩選
with st.spinner("正在即時更新台股上市櫃與 ETF 清單..."):
    ALL_STOCKS_DATA = fetch_taiwan_stocks()

# 自動生成後續主程式需要的所有格式
ALL_STOCKS = {item['id']: item['name'] for item in ALL_STOCKS_DATA}
ALL_STOCKS_LIST = [f"{item['id']} {item['name']}" for item in ALL_STOCKS_DATA]


# ==============================================================================
# 2. 你的主程式核心邏輯與介面 (無縫接軌)
# ==============================================================================
def get_stock_display_name(stock_id):
    """根據代碼取得顯示名稱，支援未知代碼"""
    name = ALL_STOCKS.get(stock_id)
    return f"{stock_id} {name}" if name else stock_id

# 介面大標題
st.title("📊 台灣股市即時決策儀表板")

# 股票搜尋與選擇下拉選單 (這裡直接帶入自動抓取的 ALL_STOCKS_LIST)
selected_stock_str = st.selectbox(
    "請選擇或輸入股票代碼/名稱：",
    options=ALL_STOCKS_LIST,
    index=ALL_STOCKS_LIST.index("2330 台積電") if "2330 台積電" in ALL_STOCKS_LIST else 0
)

# 拆分出使用者最終選取的代碼
selected_stock_id = selected_stock_str.split(" ")[0]

st.success(f"目前選擇的分析目標：{get_stock_display_name(selected_stock_id)}")


# --- 3. 技術指標計算 ---
def calculate_indicators(df):
    df = df.copy().astype(float)
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    df['RSI5'] = 100 - (100 / (1 + (gain.rolling(5).mean() / loss.rolling(5).mean())))
    df['RSI10'] = 100 - (100 / (1 + (gain.rolling(10).mean() / loss.rolling(10).mean())))
    
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    return df

# --- 4. 核心買賣訊號邏輯 ---
def get_signal_markers(df):
    buy_markers = np.full(len(df), np.nan)
    sell_markers = np.full(len(df), np.nan)
    sell_reasons = [""] * len(df)
    in_position = False 
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_position:
            if (prev_row['MA5'] <= prev_row['MA10'] and row['MA5'] > row['MA10']) and (row['RSI5'] > 50):
                buy_markers[i] = row['Close']
                in_position = True
        elif in_position:
            cond_rsi = (row['RSI5'] < row['RSI10'])
            cond_price = (row['Close'] < row['MA5'])
            cond_kdj = (row['K'] < row['D'])
            
            if cond_rsi and cond_price and cond_kdj:
                sell_markers[i] = row['Close']
                sell_reasons[i] = "RSI死叉 + KDJ死叉 + 跌破MA5"
                in_position = False 
    return buy_markers, sell_markers, sell_reasons

# --- 5. 主畫面最上方：股票搜尋欄（改移至此處，免去手機折疊問題） ---
st.title("🚀 股票買賣時機觀測")

if "final_target_code" not in st.session_state:
    st.session_state.final_target_code = "2330"

selected_stock_str = st.selectbox(
    "請輸入或選擇股票代碼：",
    options=ALL_STOCKS_LIST,
    index=list(sorted(ALL_STOCKS.keys())).index(st.session_state.final_target_code) if st.session_state.final_target_code in ALL_STOCKS else 0
)

if selected_stock_str:
    st.session_state.final_target_code = selected_stock_str.split(" ")[0].strip()

final_target_code = st.session_state.final_target_code

# --- 6. 觀測週期選擇區塊（2 排按鈕） ---
if 'view_days' not in st.session_state:
    st.session_state.view_days = 60

st.write("### 觀測週期選擇")

# 第一排按鈕 (10天、20天、30天)
row1_cols = st.columns(3)
if row1_cols[0].button("10天", use_container_width=True):
    st.session_state.view_days = 10
if row1_cols[1].button("20天", use_container_width=True):
    st.session_state.view_days = 20
if row1_cols[2].button("30天", use_container_width=True):
    st.session_state.view_days = 30

# 第二排按鈕 (60天、120天、240天)
row2_cols = st.columns(3)
if row2_cols[0].button("60天", use_container_width=True):
    st.session_state.view_days = 60
if row2_cols[1].button("120天", use_container_width=True):
    st.session_state.view_days = 120
if row2_cols[2].button("240天", use_container_width=True):
    st.session_state.view_days = 240


# --- 7. 資料抓取與圖表渲染 ---
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol):
    target_sym = f"{symbol}.TW" if "." not in symbol else symbol
    data = yf.download(target_sym, period="2y", auto_adjust=False)
    if data.empty and ".TW" in target_sym:
        target_sym = target_sym.replace(".TW", ".TWO")
        data = yf.download(target_sym, period="2y", auto_adjust=False)
    return data, target_sym

if final_target_code:
    data, final_symbol = fetch_stock_data(final_target_code)

    if not data.empty:
        display_title = get_stock_display_name(final_symbol)
        st.subheader(f"📈 {display_title}")
        
        if hasattr(data.columns, 'levels') or ('MultiIndex' in type(data.columns).__name__):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(c).strip().capitalize() for c in data.columns]
        
        df = calculate_indicators(data)
        df['買入點'], df['賣出點'], df['賣出原因'] = get_signal_markers(df)
        
        view_df = df.tail(st.session_state.view_days).copy()
        view_df['日期顯示'] = view_df.index.strftime('%Y-%m-%d')
        view_df['月份'] = view_df.index.strftime('%Y-%m')

        tick_vals = []
        tick_texts = []
        days = st.session_state.view_days

        if days == 10:
            tick_vals = view_df['日期顯示'].tolist()
            tick_texts = view_df['日期顯示'].tolist()
        elif days == 20:
            tick_vals = view_df['日期顯示'].tolist()[::5]
            tick_texts = view_df['日期顯示'].tolist()[::5]
        elif days == 30:
            tick_vals = view_df['日期顯示'].tolist()[::10]
            tick_texts = view_df['日期顯示'].tolist()[::10]
        else: 
            first_days = view_df.groupby('月份').head(1)
            tick_vals = first_days['日期顯示'].tolist()
            tick_texts = first_days['日期顯示'].tolist()

        fig = go.Figure()
        
        # 繪製實價線
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['Close'],
            mode='lines', name='實價',
            line=dict(color='#FFFFFF', width=2),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 標註買入點
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['買入點'],
            mode='markers', name='買入',
            marker=dict(symbol='triangle-up', size=14, color='#FF3131'),
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<extra></extra>"
        ))
        
        # 標註賣出點
        fig.add_trace(go.Scatter(
            x=view_df['日期顯示'], y=view_df['賣出點'],
            mode='markers', name='賣出',
            marker=dict(symbol='circle', size=14, color='#39FF14'),
            customdata=view_df['賣出原因'],
            hovertemplate="日期: %{x}<br>價格: %{y:.2f}<br><b>滿足條件: %{customdata}</b><extra></extra>"
        ))
        
        fig.update_layout(
            height=500,
            xaxis_rangeslider_visible=False,
            hovermode='closest',
            xaxis=dict(
                type='category',
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_texts,
                tickangle=45,
                fixedrange=True,
                title=""
            ),
            yaxis=dict(autorange=True, fixedrange=True, title="實價", tickformat='.2f'),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.warning("⚠️ 賣出條件：RSI死叉 + KDJ死叉 + 跌破5日線 (三重共振成立)")
    else:
        st.error("無法取得該股票的市場真實價格資料，請確認代號是否正確。")
