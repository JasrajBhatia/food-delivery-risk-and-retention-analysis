import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Intelligence",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background-color: #080D18; color: #E2E8F0; }

[data-testid="stSidebar"] { background-color: #0C1220; border-right: 1px solid #1A2540; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #64748B !important; font-size: 0.72rem !important; letter-spacing: 0.08em; text-transform: uppercase; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 2rem; max-width: 1400px; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #FFFFFF !important; }
p { color: #94A3B8; line-height: 1.6; }

.kpi-card {
    background: linear-gradient(145deg, #0F1828 0%, #141E30 100%);
    border: 1px solid #1E2D45;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 0.4rem;
    white-space: nowrap;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.1rem, 2vw, 2rem);
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
    white-space: nowrap;
}
.kpi-delta {
    font-size: 0.72rem;
    margin-top: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-delta.up { color: #34D399; }
.kpi-delta.neutral { color: #94A3B8; }
.kpi-delta.down { color: #F87171; }

.section-header {
    display: flex;
    align-items: baseline;
    gap: 0.8rem;
    margin: 2rem 0 0.3rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #1A2540;
}
.section-eyebrow {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3B82F6;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #FFFFFF;
}

.insight-box {
    background: #0A1628;
    border-left: 3px solid #3B82F6;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0 1rem 0;
    font-size: 0.8rem;
    color: #94A3B8;
    line-height: 1.5;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-green  { background: #052E16; color: #34D399; border: 1px solid #166534; }
.badge-yellow { background: #1C1400; color: #FBBF24; border: 1px solid #854D0E; }
.badge-red    { background: #1C0505; color: #F87171; border: 1px solid #7F1D1D; }
.badge-blue   { background: #0C1A3A; color: #60A5FA; border: 1px solid #1E3A6E; }

.pred-card {
    background: linear-gradient(145deg, #0F1828, #141E30);
    border: 1px solid #1E2D45;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
}
.pred-score {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.5rem 0;
}
.pred-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748B;
}

.prog-bg {
    background: #1A2540;
    border-radius: 100px;
    height: 6px;
    margin: 0.8rem 0;
    overflow: hidden;
}
.prog-fill { height: 100%; border-radius: 100px; }

.feat-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.7rem; }
.feat-name { font-size: 0.78rem; color: #94A3B8; width: 180px; flex-shrink: 0; }
.feat-bar-bg { flex: 1; background: #1A2540; border-radius: 100px; height: 5px; overflow: hidden; }
.feat-bar-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #3B82F6, #93C5FD); }
.feat-val { font-size: 0.75rem; color: #64748B; width: 45px; text-align: right; flex-shrink: 0; }

.input-card {
    background: #0F1828;
    border: 1px solid #1A2540;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}
.input-card-title {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3B82F6;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1A2540;
}

div[data-baseweb="select"] > div {
    background-color: #0F1828 !important;
    border-color: #1E2D45 !important;
    color: #E2E8F0 !important;
}
.stSelectbox label { color: #94A3B8 !important; font-size: 0.82rem !important; }

.page-hero { margin-bottom: 2rem; }
.page-hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #3B82F6;
    margin-bottom: 0.4rem;
}
.page-hero h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.3rem 0 !important;
    line-height: 1.15 !important;
}
.page-hero-sub { font-size: 0.9rem; color: #64748B; margin: 0; }

hr { border-color: #1A2540 !important; margin: 1.5rem 0 !important; }

.info-box {
    background: #0C1A3A;
    border: 1px solid #1E3A6E;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.83rem;
    color: #93C5FD;
    margin-bottom: 1rem;
}

.profile-row {
    display: flex;
    justify-content: space-between;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1A2540;
}
.profile-key { color: #64748B; font-size: 0.8rem; }
.profile-val { color: #E2E8F0; font-size: 0.8rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────
PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8", size=12),
    colorway=["#3B82F6","#60A5FA","#34D399","#FBBF24","#F87171","#A78BFA","#FB923C","#38BDF8"],
    margin=dict(l=10, r=10, t=40, b=10)
)

GRID = dict(gridcolor="#1A2540", linecolor="#1A2540", tickcolor="#1A2540", color="#94A3B8")

def apply_theme(fig, height=300, legend_h=False, title=""):
    update = {**PT, "height": height}
    if title:
        update["title"] = dict(text=title, font=dict(color="#FFFFFF", size=13))
    if legend_h:
        update["legend"] = dict(orientation="h", y=1.12, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)")
    else:
        update["legend"] = dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**update)
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    return fig

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_main():
    import os
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "uae_food_delivery_sample.csv"))
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_month"] = df["order_date"].dt.month
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    df["month_name"] = df["order_month"].map({i+1: m for i, m in enumerate(MONTHS)})
    return df

@st.cache_data
def load_users():
    return pd.read_csv("user_features.csv")

@st.cache_data
def load_orders():
    return pd.read_csv("order_features.csv")

@st.cache_data
def load_restaurants():
    return pd.read_csv("restaurant_features.csv")

@st.cache_data
def train_churn(df_u):
    d = df_u.copy().drop(columns=["user_id"])
    le = LabelEncoder()
    for c in ["city","payment_method","top_cuisine"]:
        d[c] = le.fit_transform(d[c])
    X, y = d.drop(columns=["churn_risk"]), d["churn_risk"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      random_state=42, eval_metric="logloss", use_label_encoder=False)
    m.fit(Xtr, ytr)
    return m, list(X.columns)

@st.cache_data
def train_order_risk(df_o):
    d = df_o.copy()
    d["high_risk"] = (d["order_quality_risk_score"] > 0.5).astype(int)
    d = d.drop(columns=["order_id", "order_status", "order_quality_risk_score"])
    
    # Drop risk_band if it exists as it is categorical and not needed for training
    if "risk_band" in d.columns:
        d = d.drop(columns=["risk_band"])
    
    le = LabelEncoder()
    for c in ["cuisine","city","area","traffic_level","driver_vehicle","driver_availability","payment_method"]:
        d[c] = le.fit_transform(d[c])
    X, y = d.drop(columns=["high_risk"]), d["high_risk"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      random_state=42, eval_metric="logloss", use_label_encoder=False)
    m.fit(Xtr, ytr)
    return m, list(X.columns)



@st.cache_data
def train_rest_health(df_r):
    d = df_r.copy()
    d["is_unhealthy"] = (d["restaurant_health_score"] < 0.6).astype(int)
    d = d.drop(columns=["restaurant_id","restaurant_name","top_ordered_item","restaurant_health_score"])
    le = LabelEncoder()
    for c in ["cuisine","city","area"]:
        d[c] = le.fit_transform(d[c])
    X, y = d.drop(columns=["is_unhealthy"]), d["is_unhealthy"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                      random_state=42, eval_metric="logloss", use_label_encoder=False)
    m.fit(Xtr, ytr)
    return m, list(X.columns)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def kpi(label, value, delta=None, dtype="neutral"):
    d = f"<div class='kpi-delta {dtype}'>{delta}</div>" if delta else ""
    return f"<div class='kpi-card'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>{d}</div>"

def section(eye, title):
    st.markdown(f"<div class='section-header'><span class='section-eyebrow'>{eye}</span><span class='section-title'>{title}</span></div>", unsafe_allow_html=True)

def insight(text):
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)

def badge(text, color):
    cls = {"green":"badge-green","yellow":"badge-yellow","red":"badge-red","blue":"badge-blue"}
    bc = cls.get(color, "badge-blue")
    return f"<span class='badge {bc}'>{text}</span>"

def profile_row(k, v):
    return f"<div class='profile-row'><span class='profile-key'>{k}</span><span class='profile-val'>{v}</span></div>"

def feat_bar(fname, fval, max_val):
    pct = int((fval / max_val) * 100) if max_val > 0 else 0
    return f"""<div class='feat-row'>
        <div class='feat-name'>{fname}</div>
        <div class='feat-bar-bg'><div class='feat-bar-fill' style='width:{pct}%;'></div></div>
        <div class='feat-val'>{fval:.3f}</div>
    </div>"""

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading platform data..."):
    df      = load_main()
    df_u    = load_users()
    df_o    = load_orders()
    df_r    = load_restaurants()

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0.5rem 0 1.5rem 0;'>
        <div style='font-family:Space Grotesk;font-size:1.25rem;font-weight:700;color:#FFF;'>🛵 Food Delivery</div>
        <div style='font-size:0.68rem;color:#64748B;letter-spacing:0.1em;text-transform:uppercase;margin-top:0.2rem;'>Intelligence Platform</div>
    </div>
    <hr style='border-color:#1A2540;margin-bottom:1.5rem;'>
    """, unsafe_allow_html=True)

    page = st.selectbox("NAVIGATE",
        ["Platform Overview","Churn Intelligence","Order Quality Risk","Restaurant Health"],
        label_visibility="visible")

    st.markdown("<hr style='border-color:#1A2540;margin:1.5rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem;color:#64748B;line-height:1.9;'>
        <div style='color:#94A3B8;font-weight:600;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>Dataset</div>
        771,000 Orders · 30,000 Users<br>2,000 Restaurants · 4 UAE Cities<br>52 Areas · 12 Cuisine Types<br>Jan – Dec 2024
        <br><br>
        <div style='color:#94A3B8;font-weight:600;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>Models</div>
        XGBoost · LSTM (PyTorch)<br>Random Forest · SVM<br>Logistic Regression
        <br><br>
        <div style='color:#94A3B8;font-weight:600;font-size:0.68rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem;'>Stack</div>
        Python · PySpark · Databricks<br>MLflow · SHAP · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 1: PLATFORM OVERVIEW
# ═════════════════════════════════════════════════════════════
if page == "Platform Overview":

    st.markdown("""
    <div class='page-hero'>
        <div class='page-hero-eyebrow'>Platform Intelligence</div>
        <h1>Overview</h1>
        <p class='page-hero-sub'>A full-year view of platform performance across orders, revenue, users, and delivery operations.</p>
    </div>""", unsafe_allow_html=True)

    total_orders  = len(df)
    total_rev     = df["total_price_aed"].sum()
    avg_ov        = df["total_price_aed"].mean()
    cancel_rate   = (df["order_status"] == "Cancelled").mean()
    unique_users  = df["user_id"].nunique()
    unique_rests  = df["restaurant_id"].nunique()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: st.markdown(kpi("Total Orders", f"{total_orders/1000:.0f}K"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Total Revenue", f"{total_rev/1e6:.1f}M AED"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Order Value", f"{avg_ov:.0f} AED"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Cancellation Rate", f"{cancel_rate:.1%}", "Platform wide", "neutral"), unsafe_allow_html=True)
    with c5: st.markdown(kpi("Active Users", f"{unique_users/1000:.0f}K"), unsafe_allow_html=True)
    with c6: st.markdown(kpi("Restaurants", f"{unique_rests:,}"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Monthly trend
    section("Trend", "Monthly Order Volume and Revenue")
    monthly = df.groupby("order_month").agg(orders=("order_id","count"), revenue=("total_price_aed","sum")).reset_index()
    monthly["month_name"] = pd.Categorical(monthly["order_month"].map({i+1:m for i,m in enumerate(MONTHS)}), categories=MONTHS, ordered=True)
    monthly = monthly.sort_values("month_name")

    fig_m = make_subplots(specs=[[{"secondary_y":True}]])
    fig_m.add_trace(go.Bar(x=monthly["month_name"], y=monthly["orders"], name="Orders", marker_color="#3B82F6", opacity=0.85), secondary_y=False)
    fig_m.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["revenue"], name="Revenue (AED)", mode="lines+markers", line=dict(color="#34D399",width=2), marker=dict(size=6)), secondary_y=True)
    apply_theme(fig_m, height=320, legend_h=True)
    fig_m.update_yaxes(title_text="Orders", secondary_y=False, **GRID)
    fig_m.update_yaxes(title_text="Revenue (AED)", secondary_y=True, **GRID)
    st.plotly_chart(fig_m, use_container_width=True)
    insight("📈 March shows the Ramadan surge in both order volume and revenue. February is slightly lower due to fewer calendar days. Order volume is consistent across the year with no major seasonal dips outside Ramadan.")

    # City and cuisine
    section("Geography & Cuisine", "Orders and Revenue by City and Cuisine")
    col1, col2 = st.columns(2)

    with col1:
        city_s = df.groupby("city").agg(orders=("order_id","count"), avg_value=("total_price_aed","mean")).reset_index().sort_values("orders", ascending=False)
        fig_city = px.bar(city_s, x="city", y="orders", color="avg_value", color_continuous_scale="Blues",
                          labels={"orders":"Orders","city":"","avg_value":"Avg Order (AED)"}, title="Orders by City")
        fig_city.update_layout(**PT, height=300, title=dict(text="Orders by City", font=dict(color="#FFFFFF", size=13)),
                                coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_city.update_xaxes(**GRID)
        fig_city.update_yaxes(**GRID)
        st.plotly_chart(fig_city, use_container_width=True)
        insight("🏙️ Dubai drives 60% of all orders. Abu Dhabi ranks second in both volume and average order value. Sharjah and Ajman reflect budget-conscious ordering with lower average spend per order.")

    with col2:
        cuis_s = df.groupby("cuisine").agg(orders=("order_id","count"), revenue=("total_price_aed","sum")).reset_index().sort_values("orders", ascending=False).head(10)
        fig_cuis = px.bar(cuis_s, x="orders", y="cuisine", orientation="h", color="revenue", color_continuous_scale="Blues",
                          labels={"orders":"Orders","cuisine":"","revenue":"Revenue (AED)"}, title="Top 10 Cuisines by Volume")
        fig_cuis.update_layout(**PT, height=300, title=dict(text="Top 10 Cuisines by Volume", font=dict(color="#FFFFFF", size=13)),
                                yaxis=dict(autorange="reversed", **GRID), xaxis=dict(**GRID),
                                coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        st.plotly_chart(fig_cuis, use_container_width=True)
        insight("🍛 Indian cuisine dominates reflecting UAE's South Asian demographic majority. Arabic and Pakistani follow closely. Italian and Japanese generate the highest revenue per order despite lower volumes.")

    # Hourly and status
    section("Behaviour", "Ordering Patterns by Hour and Order Status")
    col3, col4 = st.columns(2)

    with col3:
        hourly = df.groupby("order_hour")["order_id"].count().reset_index()
        hourly.columns = ["hour","orders"]
        fig_hr = px.area(hourly, x="hour", y="orders", title="Orders by Hour of Day",
                         labels={"hour":"Hour","orders":"Orders"})
        fig_hr.update_traces(line_color="#3B82F6", fillcolor="rgba(59,130,246,0.15)")
        fig_hr.update_layout(**PT, height=280, title=dict(text="Orders by Hour of Day", font=dict(color="#FFFFFF", size=13)))
        fig_hr.update_xaxes(**GRID)
        fig_hr.update_yaxes(**GRID)
        st.plotly_chart(fig_hr, use_container_width=True)
        insight("🕗 Peak ordering window is 7pm to 9pm accounting for the largest share of daily orders. The dead zone is 3am to 5am. Lunch orders spike between 12pm and 2pm for the office-going population.")

    with col4:
        status_c = df["order_status"].value_counts().reset_index()
        status_c.columns = ["status","count"]
        fig_st = px.pie(status_c, values="count", names="status", title="Order Status Distribution",
                        color_discrete_sequence=["#34D399","#F87171","#FBBF24"], hole=0.55)
        fig_st.update_layout(**PT, height=280, title=dict(text="Order Status Distribution", font=dict(color="#FFFFFF", size=13)),
                              legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
        fig_st.update_traces(textfont_color="#FFFFFF")
        st.plotly_chart(fig_st, use_container_width=True)
        insight("✅ 84.9% of orders are delivered successfully. The 11.1% cancellation rate is within acceptable industry benchmarks for food delivery platforms. In Transit orders represent active deliveries at any given time.")

    # Ramadan
    section("Seasonal", "Ramadan vs Non-Ramadan Ordering Behaviour")
    ram = df[df["is_ramadan_period"]==1]
    non_ram = df[df["is_ramadan_period"]==0]

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        uplift = (len(ram) - len(non_ram)/11) / (len(non_ram)/11) * 100
        st.markdown(kpi("Ramadan Orders", f"{len(ram)/1000:.1f}K", f"↑ {uplift:.1f}% vs avg month", "up"), unsafe_allow_html=True)
    with rc2:
        diff = ram["total_price_aed"].mean() - non_ram["total_price_aed"].mean()
        st.markdown(kpi("Ramadan Avg Order", f"{ram['total_price_aed'].mean():.0f} AED",
                        f"{'↑' if diff>0 else '↓'} {abs(diff):.1f} AED vs rest of year",
                        "up" if diff>0 else "down"), unsafe_allow_html=True)
    with rc3:
        rc = (ram["order_status"]=="Cancelled").mean()
        nrc = (non_ram["order_status"]=="Cancelled").mean()
        st.markdown(kpi("Ramadan Cancel Rate", f"{rc:.1%}", f"vs {nrc:.1%} rest of year",
                        "down" if rc>nrc else "up"), unsafe_allow_html=True)

    ram_h = ram.groupby("order_hour")["order_id"].count().reset_index()
    ram_h.columns = ["hour","orders"]
    non_h = non_ram.groupby("order_hour")["order_id"].count().reset_index()
    non_h.columns = ["hour","orders"]

    fig_ram = go.Figure()
    fig_ram.add_trace(go.Scatter(x=non_h["hour"], y=non_h["orders"], name="Regular", mode="lines", line=dict(color="#3B82F6",width=2)))
    fig_ram.add_trace(go.Scatter(x=ram_h["hour"], y=ram_h["orders"], name="Ramadan", mode="lines", line=dict(color="#FBBF24",width=2,dash="dot")))
    fig_ram.update_layout(**PT, height=280, title=dict(text="Hourly Pattern: Ramadan vs Regular", font=dict(color="#FFFFFF",size=13)),
                           legend=dict(orientation="h", y=1.12, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    fig_ram.update_xaxes(**GRID)
    fig_ram.update_yaxes(**GRID)
    st.plotly_chart(fig_ram, use_container_width=True)
    insight("🌙 Ramadan clearly shifts the peak ordering window later into the evening. The Iftar rush appears strongly at 7pm. Late night Suhoor orders between 11pm and 1am are distinctly higher during Ramadan compared to regular months.")

    # Payment and subscription
    section("Commercial", "Payment Methods and Subscription Analysis")
    cp1, cp2 = st.columns(2)

    with cp1:
        pay_s = df.groupby("payment_method").agg(orders=("order_id","count"), revenue=("total_price_aed","sum")).reset_index().sort_values("orders", ascending=False)
        fig_pay = px.bar(pay_s, x="payment_method", y="orders", color="revenue", color_continuous_scale="Blues",
                         title="Payment Method Distribution",
                         labels={"payment_method":"","orders":"Orders","revenue":"Revenue (AED)"})
        fig_pay.update_layout(**PT, height=280, title=dict(text="Payment Method Distribution", font=dict(color="#FFFFFF",size=13)),
                               coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_pay.update_xaxes(**GRID)
        fig_pay.update_yaxes(**GRID)
        st.plotly_chart(fig_pay, use_container_width=True)
        insight("💳 Credit card leads as the most used payment method. Apple Pay is strong in premium areas like Marina, DIFC, and Downtown. In-App Wallet usage reflects loyal returning users who top up for convenience.")

    with cp2:
        sub_s = df.groupby("user_subscription").agg(
            orders=("order_id","count"),
            avg_value=("total_price_aed","mean")
        ).reset_index()
        sub_s["label"] = sub_s["user_subscription"].map({1:"Subscribed",0:"Not Subscribed"})

        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(x=sub_s["label"], y=sub_s["orders"], name="Orders", marker_color="#3B82F6"))
        fig_sub.add_trace(go.Scatter(x=sub_s["label"], y=sub_s["avg_value"],
                                      name="Avg Order Value (AED)",
                                      mode="markers+text",
                                      marker=dict(size=14, color="#34D399"),
                                      text=sub_s["avg_value"].round(0).astype(int).astype(str),
                                      textposition="top center",
                                      textfont=dict(color="#34D399", size=12),
                                      yaxis="y2"))
        fig_sub.update_layout(**PT, height=280,
                               title=dict(text="Subscribed vs Non-Subscribed Users", font=dict(color="#FFFFFF",size=13)),
                               yaxis=dict(**GRID),
                               yaxis2=dict(overlaying="y", side="right", **GRID),
                               legend=dict(orientation="h", y=1.15, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
        fig_sub.update_xaxes(**GRID)
        st.plotly_chart(fig_sub, use_container_width=True)
        insight("📊 Subscribed users place significantly more orders than non-subscribed users. The average order value is nearly identical between both groups, suggesting subscription drives frequency rather than spend per order.")

    # Top areas
    section("Geography", "Top 15 Areas by Order Volume")
    area_s = df.groupby(["city","area"]).agg(orders=("order_id","count"), revenue=("total_price_aed","sum")).reset_index().sort_values("orders",ascending=False).head(15)
    fig_area = px.bar(area_s, x="orders", y="area", color="city", orientation="h",
                      labels={"orders":"Orders","area":"","city":"City"},
                      color_discrete_sequence=["#3B82F6","#34D399","#FBBF24","#F87171"])
    fig_area.update_layout(**PT, height=420,
                            title=dict(text="Top 15 Areas by Order Volume", font=dict(color="#FFFFFF",size=13)),
                            yaxis=dict(autorange="reversed", **GRID), xaxis=dict(**GRID),
                            legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_area, use_container_width=True)
    insight("📍 Marina, Downtown, and JLT are the three highest volume areas in Dubai. Al Nahda in Sharjah ranks surprisingly high due to dense residential population. Deira leads among budget areas reflecting high footfall and affordable dining options.")

    # Traffic and delivery
    section("Operations", "Delivery Performance by Traffic and Distance")
    co1, co2 = st.columns(2)

    with co1:
        traf_s = df.groupby("traffic_level").agg(
            orders=("order_id","count"),
            avg_dur=("delivery_duration_mins","mean"),
            cancel=("order_status", lambda x: (x=="Cancelled").mean()*100)
        ).reset_index()
        traf_order = ["Low","Medium","High","Very High"]
        traf_s["traffic_level"] = pd.Categorical(traf_s["traffic_level"], categories=traf_order, ordered=True)
        traf_s = traf_s.sort_values("traffic_level")

        fig_traf = px.bar(traf_s, x="traffic_level", y="avg_dur", color="cancel",
                          color_continuous_scale="Reds",
                          labels={"traffic_level":"","avg_dur":"Avg Delivery Time (min)","cancel":"Cancel Rate %"},
                          title="Avg Delivery Time by Traffic Level")
        fig_traf.update_layout(**PT, height=280,
                                title=dict(text="Avg Delivery Time by Traffic Level", font=dict(color="#FFFFFF",size=13)),
                                coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_traf.update_xaxes(**GRID)
        fig_traf.update_yaxes(**GRID)
        st.plotly_chart(fig_traf, use_container_width=True)
        insight("🚦 Very High traffic more than doubles average delivery time compared to Low traffic. Cancellation rate also rises sharply under heavy traffic as customers lose patience with delayed ETAs.")

    with co2:
        veh_s = df.groupby("driver_vehicle").agg(
            orders=("order_id","count"),
            avg_dur=("delivery_duration_mins","mean")
        ).reset_index()
        fig_veh = px.bar(veh_s, x="driver_vehicle", y="avg_dur", color="orders",
                         color_continuous_scale="Blues",
                         labels={"driver_vehicle":"","avg_dur":"Avg Delivery Time (min)","orders":"Orders"},
                         title="Avg Delivery Time by Vehicle Type")
        fig_veh.update_layout(**PT, height=280,
                               title=dict(text="Avg Delivery Time by Vehicle Type", font=dict(color="#FFFFFF",size=13)),
                               coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_veh.update_xaxes(**GRID)
        fig_veh.update_yaxes(**GRID)
        st.plotly_chart(fig_veh, use_container_width=True)
        insight("🛵 Motorcycles handle the vast majority of deliveries and are faster than bicycles on average. Bicycles are slower but suited for short distance orders in dense areas like Downtown and Marina.")


# ═════════════════════════════════════════════════════════════
# PAGE 2: CHURN INTELLIGENCE
# ═════════════════════════════════════════════════════════════
elif page == "Churn Intelligence":

    st.markdown("""
    <div class='page-hero'>
        <div class='page-hero-eyebrow'>Model 01 · XGBoost · 94.1% Accuracy · ROC AUC 0.979</div>
        <h1>Churn Intelligence</h1>
        <p class='page-hero-sub'>Identify users at risk of leaving the platform. Understand why they churn and act before they do.</p>
    </div>""", unsafe_allow_html=True)

    churners = df_u[df_u["churn_risk"]==1]
    loyals   = df_u[df_u["churn_risk"]==0]
    churn_rate = df_u["churn_risk"].mean()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Users at Churn Risk", f"{len(churners):,}", f"{churn_rate:.1%} of all users","down"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("At-Risk Revenue", f"{churners['total_spend'].sum()/1e6:.1f}M AED", "Combined lifetime spend","down"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Orders (Churners)", f"{churners['total_orders'].mean():.1f}", f"vs {loyals['total_orders'].mean():.1f} for loyal users","down"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Loyal Users", f"{len(loyals):,}", f"{1-churn_rate:.1%} of all users","up"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Churn by city and cuisine
    section("Analysis", "Churn Rate by City and Preferred Cuisine")
    cc1, cc2 = st.columns(2)

    with cc1:
        city_ch = df_u.groupby("city").agg(total=("user_id","count"), churned=("churn_risk","sum")).reset_index()
        city_ch["churn_rate"] = city_ch["churned"] / city_ch["total"]
        city_ch = city_ch.sort_values("churn_rate", ascending=False)
        fig_cch = px.bar(city_ch, x="city", y="churn_rate", color="churn_rate",
                         color_continuous_scale="Reds",
                         labels={"city":"","churn_rate":"Churn Rate"})
        fig_cch.update_layout(**PT, height=300,
                               title=dict(text="Churn Rate by City", font=dict(color="#FFFFFF",size=13)),
                               yaxis=dict(tickformat=".0%", **GRID), xaxis=dict(**GRID),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_cch, use_container_width=True)
        insight("🏙️ Churn rates are fairly consistent across cities reflecting that churn is driven by user behaviour patterns rather than geography. Small differences may reflect variations in restaurant quality and delivery times per city.")

    with cc2:
        cuis_ch = df_u.groupby("top_cuisine").agg(total=("user_id","count"), churned=("churn_risk","sum")).reset_index()
        cuis_ch["churn_rate"] = cuis_ch["churned"] / cuis_ch["total"]
        cuis_ch = cuis_ch.sort_values("churn_rate", ascending=False)
        fig_ccu = px.bar(cuis_ch, x="churn_rate", y="top_cuisine", orientation="h",
                         color="churn_rate", color_continuous_scale="Reds",
                         labels={"top_cuisine":"","churn_rate":"Churn Rate"})
        fig_ccu.update_layout(**PT, height=300,
                               title=dict(text="Churn Rate by Preferred Cuisine", font=dict(color="#FFFFFF",size=13)),
                               xaxis=dict(tickformat=".0%", **GRID),
                               yaxis=dict(autorange="reversed", **GRID),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_ccu, use_container_width=True)
        insight("🍜 Cuisine preference alone is not a strong predictor of churn. Differences across cuisines are minor, confirming the SHAP analysis finding that what cuisine a user orders matters far less than how frequently they order.")

    # Radar comparison
    section("Profiles", "Churner vs Loyal User Behaviour Comparison")
    metrics = ["total_orders","avg_order_value","total_spend","cancellation_rate_pct","avg_delivery_duration","ramadan_orders"]
    labels  = ["Total Orders","Avg Order Value","Total Spend","Cancel Rate %","Avg Delivery Time","Ramadan Orders"]
    cm = churners[metrics].mean()
    lm = loyals[metrics].mean()

    fig_rad = go.Figure()
    fig_rad.add_trace(go.Scatterpolar(
        r=[cm[m]/max(cm[m],lm[m]) for m in metrics], theta=labels,
        fill="toself", name="Churners",
        line_color="#F87171", fillcolor="rgba(248,113,113,0.15)"))
    fig_rad.add_trace(go.Scatterpolar(
        r=[lm[m]/max(cm[m],lm[m]) for m in metrics], theta=labels,
        fill="toself", name="Loyal Users",
        line_color="#34D399", fillcolor="rgba(52,211,153,0.15)"))
    fig_rad.update_layout(**PT, height=380,
                           polar=dict(bgcolor="rgba(0,0,0,0)",
                                      radialaxis=dict(visible=True, gridcolor="#1A2540", color="#64748B"),
                                      angularaxis=dict(gridcolor="#1A2540", color="#94A3B8")),
                           legend=dict(orientation="h", y=-0.1, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_rad, use_container_width=True)
    insight("📊 The radar chart clearly separates churners from loyal users. Loyal users show significantly higher total orders, total spend, and Ramadan engagement. Churners have slightly higher cancellation rates, suggesting poor delivery experiences contribute to churn.")

    # Spend distribution
    section("Spend", "Lifetime Spend Distribution: Churners vs Loyal Users")
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Histogram(x=churners["total_spend"], name="Churners", marker_color="#F87171", opacity=0.7, nbinsx=40))
    fig_sp.add_trace(go.Histogram(x=loyals["total_spend"], name="Loyal", marker_color="#34D399", opacity=0.7, nbinsx=40))
    fig_sp.update_layout(**PT, height=280,
                          title=dict(text="Lifetime Spend Distribution", font=dict(color="#FFFFFF",size=13)),
                          barmode="overlay",
                          xaxis=dict(title="Total Spend (AED)", **GRID),
                          yaxis=dict(title="Users", **GRID),
                          legend=dict(orientation="h", y=1.12, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_sp, use_container_width=True)
    insight("💰 Churners are concentrated at lower lifetime spend values. Loyal users have a much wider and higher spend distribution. High value users above 2,000 AED lifetime spend are almost exclusively non-churners, making them a priority retention segment.")

    # Orders distribution
    section("Orders", "Total Orders Distribution: Churners vs Loyal Users")
    fig_ord = go.Figure()
    fig_ord.add_trace(go.Histogram(x=churners["total_orders"], name="Churners", marker_color="#F87171", opacity=0.7, nbinsx=30))
    fig_ord.add_trace(go.Histogram(x=loyals["total_orders"], name="Loyal", marker_color="#34D399", opacity=0.7, nbinsx=30))
    fig_ord.update_layout(**PT, height=280,
                           title=dict(text="Total Orders Distribution", font=dict(color="#FFFFFF",size=13)),
                           barmode="overlay",
                           xaxis=dict(title="Total Orders", **GRID),
                           yaxis=dict(title="Users", **GRID),
                           legend=dict(orientation="h", y=1.12, font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_ord, use_container_width=True)
    insight("🛒 Churners cluster heavily at low order counts (1 to 10 orders). Loyal users spread across a much wider range including many high-frequency orderers above 30 orders. Total orders is the single strongest predictor of churn as confirmed by SHAP analysis.")

    # Subscription and churn
    section("Subscription", "Churn Rate by Subscription Status")
    sub_ch = df_u.groupby("user_subscription").agg(total=("user_id","count"), churned=("churn_risk","sum")).reset_index()
    sub_ch["churn_rate"] = sub_ch["churned"] / sub_ch["total"]
    sub_ch["label"] = sub_ch["user_subscription"].map({1:"Subscribed",0:"Not Subscribed"})
    fig_sub_ch = px.bar(sub_ch, x="label", y="churn_rate", color="churn_rate",
                        color_continuous_scale="Reds",
                        labels={"label":"","churn_rate":"Churn Rate"})
    fig_sub_ch.update_layout(**PT, height=280,
                              title=dict(text="Churn Rate by Subscription Status", font=dict(color="#FFFFFF",size=13)),
                              yaxis=dict(tickformat=".0%", **GRID), xaxis=dict(**GRID),
                              coloraxis_showscale=False)
    st.plotly_chart(fig_sub_ch, use_container_width=True)
    insight("📋 Subscription status alone does not dramatically separate churners from non-churners. This suggests the subscription programme is not yet a strong enough retention mechanism and may benefit from enhanced perks or personalised incentives.")

    # Individual user lookup
    section("Prediction", "Individual User Churn Risk Lookup")
    st.markdown("<div class='info-box'>Select a user ID to see their churn probability predicted by the XGBoost model trained on real order history data.</div>", unsafe_allow_html=True)

    with st.spinner("Preparing churn model..."):
        churn_model, churn_feats = train_churn(df_u)

    sample_u = df_u.sample(min(300, len(df_u)), random_state=42)
    sel_uid = st.selectbox("Select User ID", sample_u["user_id"].tolist())
    urow = df_u[df_u["user_id"]==sel_uid].iloc[0]

    pu1, pu2 = st.columns([1, 1.2], gap="large")

    with pu1:
        st.markdown("<div class='input-card'><div class='input-card-title'>User Profile</div>", unsafe_allow_html=True)
        rows = [
            ("Total Orders", int(urow["total_orders"])),
            ("Total Spend", f"{urow['total_spend']:.2f} AED"),
            ("Avg Order Value", f"{urow['avg_order_value']:.2f} AED"),
            ("Total Cancellations", int(urow["total_cancellations"])),
            ("Cancellation Rate", f"{urow['cancellation_rate_pct']:.1f}%"),
            ("City", urow["city"]),
            ("Top Cuisine", urow["top_cuisine"]),
            ("Subscription", "Active" if urow["user_subscription"]==1 else "Inactive"),
            ("Ramadan Orders", int(urow["ramadan_orders"])),
            ("Weekend Orders", int(urow["weekend_orders"])),
            ("Avg Risk Score", f"{urow['avg_risk_score']:.3f}"),
        ]
        for k, v in rows:
            st.markdown(profile_row(k, v), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pu2:
        u_enc = df_u[df_u["user_id"]==sel_uid].copy().drop(columns=["user_id","churn_risk"])
        le = LabelEncoder()
        for c in ["city","payment_method","top_cuisine"]:
            le.fit(df_u[c].unique())
            u_enc[c] = le.transform(u_enc[c])
        u_enc = u_enc[churn_feats]
        cp = churn_model.predict_proba(u_enc)[0][1]
        actual = int(urow["churn_risk"])

        if cp < 0.35:   rc, rl, rb = "#34D399","Low Risk","green"
        elif cp < 0.65: rc, rl, rb = "#FBBF24","Medium Risk","yellow"
        else:           rc, rl, rb = "#F87171","High Risk","red"

        st.markdown(f"""
        <div class='pred-card'>
            <div class='pred-label'>Churn Probability</div>
            <div class='pred-score' style='color:{rc};'>{cp:.0%}</div>
            <div style='margin:0.6rem 0;'>{badge(rl, rb)}</div>
            <div class='prog-bg'><div class='prog-fill' style='width:{cp*100:.0f}%;background:{rc};'></div></div>
            <div style='font-size:0.78rem;color:#64748B;margin-top:0.5rem;'>
                Actual label: {'🔴 Churn Risk' if actual==1 else '🟢 No Churn'}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        imps = churn_model.feature_importances_
        fi = sorted(zip(churn_feats, imps), key=lambda x: x[1], reverse=True)[:6]
        max_i = max(f[1] for f in fi) or 1

        st.markdown("<div class='input-card'><div class='input-card-title'>Top Feature Importances</div>", unsafe_allow_html=True)
        for fn, fv in fi:
            st.markdown(feat_bar(fn, fv, max_i), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        insight(f"The model predicted {cp:.0%} churn probability for this user. {'High order frequency and spend suggest strong platform loyalty.' if cp < 0.35 else 'Low order frequency and engagement are the primary churn drivers for this user.' if cp >= 0.65 else 'This user shows mixed signals. Monitor their next 30 days ordering behaviour closely.'}")


# ═════════════════════════════════════════════════════════════
# PAGE 3: ORDER QUALITY RISK
# ═════════════════════════════════════════════════════════════
elif page == "Order Quality Risk":

    st.markdown("""
    <div class='page-hero'>
        <div class='page-hero-eyebrow'>Model 02 · XGBoost · 94.2% Accuracy · ROC AUC 0.974</div>
        <h1>Order Quality Risk</h1>
        <p class='page-hero-sub'>Score every order's risk of a poor delivery experience. Understand what drives risk and where to intervene.</p>
    </div>""", unsafe_allow_html=True)

    high_risk = df_o[df_o["order_quality_risk_score"]>0.5]
    low_risk  = df_o[df_o["order_quality_risk_score"]<=0.5]
    avg_risk  = df_o["order_quality_risk_score"].mean()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("High Risk Orders", f"{len(high_risk):,}", f"{len(high_risk)/len(df_o):.1%} of all orders","down"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Avg Risk Score", f"{avg_risk:.3f}", "Platform wide average","neutral"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Low Risk Orders", f"{len(low_risk):,}", f"{len(low_risk)/len(df_o):.1%} of all orders","up"), unsafe_allow_html=True)
    with c4:
        cancelled = df_o[df_o["order_status"]=="Cancelled"]
        st.markdown(kpi("Cancelled Orders", f"{len(cancelled):,}", f"{len(cancelled)/len(df_o):.1%} cancellation rate","down"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk by traffic and area
    section("Risk Drivers", "Risk Score by Traffic Level and Top Risk Areas")
    rt1, rt2 = st.columns(2)

    with rt1:
        traf_r = df_o.groupby("traffic_level")["order_quality_risk_score"].mean().reset_index()
        traf_r.columns = ["traffic_level","avg_risk"]
        tord = ["Low","Medium","High","Very High"]
        traf_r["traffic_level"] = pd.Categorical(traf_r["traffic_level"], categories=tord, ordered=True)
        traf_r = traf_r.sort_values("traffic_level")
        fig_tr = px.bar(traf_r, x="traffic_level", y="avg_risk", color="avg_risk", color_continuous_scale="Reds",
                        labels={"traffic_level":"","avg_risk":"Avg Risk Score"})
        fig_tr.update_layout(**PT, height=300,
                              title=dict(text="Avg Risk Score by Traffic Level", font=dict(color="#FFFFFF",size=13)),
                              coloraxis_showscale=False)
        fig_tr.update_xaxes(**GRID)
        fig_tr.update_yaxes(**GRID)
        st.plotly_chart(fig_tr, use_container_width=True)
        insight("🚦 Traffic level is the single strongest driver of order quality risk. Very High traffic produces a risk score more than double that of Low traffic. Proactive intervention during peak hours can significantly reduce poor delivery experiences.")

    with rt2:
        area_r = df_o.groupby("area")["order_quality_risk_score"].mean().reset_index()
        area_r.columns = ["area","avg_risk"]
        area_r = area_r.sort_values("avg_risk", ascending=False).head(12)
        fig_ar = px.bar(area_r, x="avg_risk", y="area", orientation="h", color="avg_risk",
                        color_continuous_scale="Reds",
                        labels={"area":"","avg_risk":"Avg Risk Score"})
        fig_ar.update_layout(**PT, height=300,
                              title=dict(text="Top 12 Areas by Avg Risk Score", font=dict(color="#FFFFFF",size=13)),
                              yaxis=dict(autorange="reversed", **GRID), xaxis=dict(**GRID),
                              coloraxis_showscale=False)
        st.plotly_chart(fig_ar, use_container_width=True)
        insight("📍 Areas with higher risk scores tend to be those with longer delivery distances or more complex road networks. These areas benefit most from proactive driver allocation and real-time traffic monitoring.")

    # Risk by hour and vehicle
    section("Patterns", "Risk by Hour of Day and Driver Vehicle")
    rh1, rh2 = st.columns(2)

    with rh1:
        hour_r = df_o.groupby("order_hour")["order_quality_risk_score"].mean().reset_index()
        hour_r.columns = ["hour","avg_risk"]
        fig_hr2 = px.area(hour_r, x="hour", y="avg_risk",
                          labels={"hour":"Hour","avg_risk":"Avg Risk Score"})
        fig_hr2.update_traces(line_color="#F87171", fillcolor="rgba(248,113,113,0.15)")
        fig_hr2.update_layout(**PT, height=280,
                               title=dict(text="Avg Risk Score by Hour of Day", font=dict(color="#FFFFFF",size=13)))
        fig_hr2.update_xaxes(**GRID)
        fig_hr2.update_yaxes(**GRID)
        st.plotly_chart(fig_hr2, use_container_width=True)
        insight("🕗 Risk peaks during evening hours coinciding with peak traffic and high order volume simultaneously. The combination of maximum demand and maximum traffic between 7pm and 9pm creates the highest risk window for poor deliveries.")

    with rh2:
        veh_r = df_o.groupby("driver_vehicle").agg(avg_risk=("order_quality_risk_score","mean"), orders=("order_quality_risk_score","count")).reset_index()
        fig_vr = px.bar(veh_r, x="driver_vehicle", y="avg_risk", color="avg_risk",
                        color_continuous_scale="Reds",
                        labels={"driver_vehicle":"","avg_risk":"Avg Risk Score"})
        fig_vr.update_layout(**PT, height=280,
                              title=dict(text="Avg Risk Score by Driver Vehicle", font=dict(color="#FFFFFF",size=13)),
                              coloraxis_showscale=False)
        fig_vr.update_xaxes(**GRID)
        fig_vr.update_yaxes(**GRID)
        st.plotly_chart(fig_vr, use_container_width=True)
        insight("🚲 Bicycles carry a higher average risk score than motorcycles as expected. Bicycles are slower and more susceptible to traffic delays particularly over longer distances. Motorcycle deliveries are faster and more predictable.")

    # Cancellation rate by risk level
    section("Cancellation", "Cancellation Rate by Risk Band")
    df_o["risk_band"] = pd.cut(df_o["order_quality_risk_score"],
                                bins=[0,0.25,0.5,0.75,1.0],
                                labels=["Very Low (0-0.25)","Low (0.25-0.5)","High (0.5-0.75)","Very High (0.75-1.0)"])
    risk_cancel = df_o.groupby("risk_band").agg(
        cancel_rate=("order_status", lambda x: (x=="Cancelled").mean()*100),
        orders=("order_quality_risk_score","count")
    ).reset_index()
    fig_rc = px.bar(risk_cancel, x="risk_band", y="cancel_rate", color="cancel_rate",
                    color_continuous_scale="Reds",
                    labels={"risk_band":"Risk Band","cancel_rate":"Cancellation Rate %"})
    fig_rc.update_layout(**PT, height=280,
                          title=dict(text="Cancellation Rate by Risk Band", font=dict(color="#FFFFFF",size=13)),
                          coloraxis_showscale=False)
    fig_rc.update_xaxes(**GRID)
    fig_rc.update_yaxes(**GRID)
    st.plotly_chart(fig_rc, use_container_width=True)
    insight("❌ Cancellation rate rises sharply as risk score increases. Very High risk orders cancel at significantly higher rates than Very Low risk orders, validating that the risk score is a meaningful predictor of real order outcomes.")

    # Risk distribution
    section("Distribution", "Order Quality Risk Score Distribution")
    fig_dist = px.histogram(df_o, x="order_quality_risk_score", nbins=50,
                             color_discrete_sequence=["#3B82F6"],
                             labels={"order_quality_risk_score":"Risk Score"})
    fig_dist.add_vline(x=0.5, line_dash="dash", line_color="#F87171",
                       annotation_text="High Risk Threshold",
                       annotation_font_color="#F87171")
    fig_dist.update_layout(**PT, height=280,
                            title=dict(text="Distribution of Order Quality Risk Scores", font=dict(color="#FFFFFF",size=13)))
    fig_dist.update_xaxes(**GRID)
    fig_dist.update_yaxes(**GRID)
    st.plotly_chart(fig_dist, use_container_width=True)
    insight("📊 The majority of orders cluster in the low to medium risk range (0.2 to 0.5). The distribution is right-skewed with a smaller but significant tail of high risk orders above 0.5. This 15.1% high risk segment is where operational intervention delivers the most value.")

    # Individual order lookup
    section("Prediction", "Individual Order Risk Lookup")
    st.markdown("<div class='info-box'>Select an order to see its risk score and the factors driving the prediction.</div>", unsafe_allow_html=True)

    with st.spinner("Preparing order risk model..."):
        order_model, order_feats = train_order_risk(df_o)

    sample_o = df_o.sample(min(300, len(df_o)), random_state=42)
    sel_oid = st.selectbox("Select Order ID", sample_o["order_id"].tolist())
    orow = df_o[df_o["order_id"]==sel_oid].iloc[0]

    po1, po2 = st.columns([1, 1.2], gap="large")

    with po1:
        st.markdown("<div class='input-card'><div class='input-card-title'>Order Details</div>", unsafe_allow_html=True)
        orows = [
            ("Cuisine", orow["cuisine"]),
            ("City", orow["city"]),
            ("Area", orow["area"]),
            ("Traffic Level", orow["traffic_level"]),
            ("Driver Vehicle", orow["driver_vehicle"]),
            ("Driver Availability", orow["driver_availability"]),
            ("Distance (km)", f"{orow['delivery_distance_km']:.2f}"),
            ("Duration (min)", f"{orow['delivery_duration_mins']:.1f}"),
            ("Order Value (AED)", f"{orow['total_price_aed']:.2f}"),
            ("Order Hour", int(orow["order_hour"])),
            ("Weekend", "Yes" if orow["is_weekend"]==1 else "No"),
            ("Order Status", orow["order_status"]),
        ]
        for k, v in orows:
            st.markdown(profile_row(k, v), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with po2:
        actual_risk = orow["order_quality_risk_score"]
        is_high = actual_risk > 0.5
        hc = "#F87171" if is_high else "#34D399"
        hl = "High Risk" if is_high else "Low Risk"
        hb = "red" if is_high else "green"

        st.markdown(f"""
        <div class='pred-card'>
            <div class='pred-label'>Risk Score</div>
            <div class='pred-score' style='color:{hc};'>{actual_risk:.3f}</div>
            <div style='margin:0.6rem 0;'>{badge(hl, hb)}</div>
            <div class='prog-bg'><div class='prog-fill' style='width:{actual_risk*100:.0f}%;background:{hc};'></div></div>
            <div style='font-size:0.78rem;color:#64748B;margin-top:0.5rem;'>
                Status: {orow['order_status']} · Traffic: {orow['traffic_level']}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        imps = order_model.feature_importances_
        fi = sorted(zip(order_feats, imps), key=lambda x: x[1], reverse=True)[:6]
        max_i = max(f[1] for f in fi) or 1

        st.markdown("<div class='input-card'><div class='input-card-title'>Top Feature Importances</div>", unsafe_allow_html=True)
        for fn, fv in fi:
            st.markdown(feat_bar(fn, fv, max_i), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        insight(f"This order has a risk score of {actual_risk:.3f}. {'Traffic level and delivery distance are the primary risk factors for this order.' if is_high else 'Low traffic and short distance make this a low risk delivery.'}")


# ═════════════════════════════════════════════════════════════
# PAGE 4: RESTAURANT HEALTH
# ═════════════════════════════════════════════════════════════
elif page == "Restaurant Health":

    st.markdown("""
    <div class='page-hero'>
        <div class='page-hero-eyebrow'>Model 03 · XGBoost · 97.2% Accuracy · ROC AUC 0.993</div>
        <h1>Restaurant Health</h1>
        <p class='page-hero-sub'>Flag underperforming restaurants before they damage platform reputation and drive users away.</p>
    </div>""", unsafe_allow_html=True)

    unhealthy = df_r[df_r["restaurant_health_score"]<0.6]
    healthy   = df_r[df_r["restaurant_health_score"]>=0.6]
    avg_h = df_r["restaurant_health_score"].mean()
    avg_c = df_r["cancellation_rate_pct"].mean()

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(kpi("Unhealthy Restaurants", f"{len(unhealthy)}", f"{len(unhealthy)/len(df_r):.1%} of all restaurants","down"), unsafe_allow_html=True)
    with c2: st.markdown(kpi("Healthy Restaurants", f"{len(healthy)}", f"{len(healthy)/len(df_r):.1%} of all restaurants","up"), unsafe_allow_html=True)
    with c3: st.markdown(kpi("Avg Health Score", f"{avg_h:.3f}", "Platform wide average","neutral"), unsafe_allow_html=True)
    with c4: st.markdown(kpi("Avg Cancel Rate", f"{avg_c:.1f}%", "Across all restaurants","neutral"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Health by city and cuisine
    section("Analysis", "Restaurant Health by City and Cuisine")
    rh1, rh2 = st.columns(2)

    with rh1:
        city_h = df_r.groupby("city").agg(
            avg_health=("restaurant_health_score","mean"),
            unhealthy=("restaurant_health_score", lambda x: (x<0.6).sum()),
            total=("restaurant_id","count")
        ).reset_index()
        city_h["unhealthy_pct"] = city_h["unhealthy"] / city_h["total"]
        fig_ch = px.bar(city_h, x="city", y="avg_health", color="unhealthy_pct",
                        color_continuous_scale="RdYlGn_r",
                        labels={"city":"","avg_health":"Avg Health Score","unhealthy_pct":"Unhealthy %"})
        fig_ch.update_layout(**PT, height=300,
                              title=dict(text="Avg Health Score by City", font=dict(color="#FFFFFF",size=13)),
                              coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_ch.update_xaxes(**GRID)
        fig_ch.update_yaxes(**GRID)
        st.plotly_chart(fig_ch, use_container_width=True)
        insight("🏙️ Health scores are relatively consistent across cities. The colour gradient showing unhealthy percentage reveals which cities have a higher concentration of underperforming restaurants requiring partnerships team attention.")

    with rh2:
        cuis_h = df_r.groupby("cuisine").agg(
            avg_health=("restaurant_health_score","mean"),
            unhealthy=("restaurant_health_score", lambda x: (x<0.6).sum()),
            total=("restaurant_id","count")
        ).reset_index()
        cuis_h["unhealthy_pct"] = cuis_h["unhealthy"] / cuis_h["total"]
        cuis_h = cuis_h.sort_values("avg_health")
        fig_cuh = px.bar(cuis_h, x="avg_health", y="cuisine", orientation="h",
                         color="unhealthy_pct", color_continuous_scale="RdYlGn_r",
                         labels={"cuisine":"","avg_health":"Avg Health Score","unhealthy_pct":"Unhealthy %"})
        fig_cuh.update_layout(**PT, height=300,
                               title=dict(text="Avg Health Score by Cuisine", font=dict(color="#FFFFFF",size=13)),
                               coloraxis_colorbar=dict(tickfont=dict(color="#94A3B8")))
        fig_cuh.update_xaxes(**GRID)
        fig_cuh.update_yaxes(**GRID)
        st.plotly_chart(fig_cuh, use_container_width=True)
        insight("🍽️ Some cuisine types show consistently lower health scores, suggesting systemic operational challenges within those restaurant categories. These cuisines may benefit from targeted operational support from the platform partnerships team.")

    # Cancellation vs health scatter
    section("Operational", "Cancellation Rate vs Health Score")
    fig_sc = px.scatter(df_r, x="cancellation_rate_pct", y="restaurant_health_score",
                        color="cuisine", size="total_orders",
                        labels={"cancellation_rate_pct":"Cancellation Rate %",
                                "restaurant_health_score":"Health Score",
                                "cuisine":"Cuisine","total_orders":"Total Orders"},
                        hover_data=["restaurant_name","city","area"])
    fig_sc.add_hline(y=0.6, line_dash="dash", line_color="#F87171",
                     annotation_text="Unhealthy Threshold",
                     annotation_font_color="#F87171")
    fig_sc.update_layout(**PT, height=380,
                          title=dict(text="Cancellation Rate vs Health Score by Cuisine", font=dict(color="#FFFFFF",size=13)),
                          legend=dict(font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"))
    fig_sc.update_xaxes(**GRID)
    fig_sc.update_yaxes(**GRID)
    st.plotly_chart(fig_sc, use_container_width=True)
    insight("📉 There is a clear negative correlation between cancellation rate and health score. Restaurants with high cancellation rates consistently fall below the 0.6 health threshold. Hover over any dot to see the specific restaurant name, city, and area.")

    # Health distribution
    section("Distribution", "Health Score Distribution Across All Restaurants")
    fig_hd = px.histogram(df_r, x="restaurant_health_score", nbins=40,
                           color_discrete_sequence=["#3B82F6"],
                           labels={"restaurant_health_score":"Health Score"})
    fig_hd.add_vline(x=0.6, line_dash="dash", line_color="#F87171",
                     annotation_text="Unhealthy Threshold (< 0.6)",
                     annotation_font_color="#F87171")
    fig_hd.update_layout(**PT, height=280,
                          title=dict(text="Distribution of Restaurant Health Scores", font=dict(color="#FFFFFF",size=13)))
    fig_hd.update_xaxes(**GRID)
    fig_hd.update_yaxes(**GRID)
    st.plotly_chart(fig_hd, use_container_width=True)
    insight("📊 The majority of restaurants cluster in the 0.6 to 0.9 health score range indicating generally good platform performance. The 22.9% falling below the 0.6 threshold represents restaurants that need immediate operational review.")

    # Delivery duration comparison
    section("Delivery", "Avg Delivery Duration: Healthy vs Unhealthy Restaurants")
    health_label = df_r.copy()
    health_label["status"] = health_label["restaurant_health_score"].apply(lambda x: "Healthy" if x>=0.6 else "Unhealthy")
    dur_comp = health_label.groupby("status")["avg_delivery_duration"].mean().reset_index()
    fig_dur = px.bar(dur_comp, x="status", y="avg_delivery_duration",
                     color="status", color_discrete_map={"Healthy":"#34D399","Unhealthy":"#F87171"},
                     labels={"status":"","avg_delivery_duration":"Avg Delivery Duration (min)"})
    fig_dur.update_layout(**PT, height=280,
                           title=dict(text="Avg Delivery Duration by Health Status", font=dict(color="#FFFFFF",size=13)),
                           showlegend=False)
    fig_dur.update_xaxes(**GRID)
    fig_dur.update_yaxes(**GRID)
    st.plotly_chart(fig_dur, use_container_width=True)
    insight("⏱️ Unhealthy restaurants have notably longer average delivery durations than healthy ones. Slow food preparation at the restaurant level cascades into longer total delivery times, increasing customer dissatisfaction and cancellation likelihood.")

    # Flagged restaurants table
    section("Flagged", "Restaurants Currently Flagged as Unhealthy")
    unhealthy_disp = unhealthy[["restaurant_name","cuisine","city","area",
                                  "restaurant_health_score","cancellation_rate_pct",
                                  "avg_delivery_duration","total_orders"]].copy()
    unhealthy_disp.columns = ["Restaurant","Cuisine","City","Area",
                               "Health Score","Cancel Rate %","Avg Delivery (min)","Total Orders"]
    unhealthy_disp = unhealthy_disp.sort_values("Health Score")
    st.dataframe(unhealthy_disp.reset_index(drop=True), use_container_width=True, height=300)
    insight(f"📋 {len(unhealthy)} restaurants are currently flagged as unhealthy. These restaurants should be prioritised for outreach by the partnerships team. The lowest scoring restaurants at the top of the table represent the most urgent cases.")

    # Individual restaurant lookup
    section("Prediction", "Individual Restaurant Health Lookup")
    st.markdown("<div class='info-box'>Select a restaurant to see its health score and the operational factors driving the prediction.</div>", unsafe_allow_html=True)

    with st.spinner("Preparing restaurant health model..."):
        rest_model, rest_feats = train_rest_health(df_r)

    sel_rest = st.selectbox("Select Restaurant", df_r["restaurant_name"].tolist())
    rrow = df_r[df_r["restaurant_name"]==sel_rest].iloc[0]

    pr1, pr2 = st.columns([1, 1.2], gap="large")

    with pr1:
        st.markdown("<div class='input-card'><div class='input-card-title'>Restaurant Profile</div>", unsafe_allow_html=True)
        rrows = [
            ("Cuisine", rrow["cuisine"]),
            ("City", rrow["city"]),
            ("Area", rrow["area"]),
            ("Total Orders", int(rrow["total_orders"])),
            ("Total Cancellations", int(rrow["total_cancellations"])),
            ("Cancellation Rate", f"{rrow['cancellation_rate_pct']:.2f}%"),
            ("Avg Delivery (min)", f"{rrow['avg_delivery_duration']:.1f}"),
            ("Avg Order Value", f"{rrow['avg_order_value']:.2f} AED"),
            ("Avg Quality Risk", f"{rrow['avg_quality_risk']:.3f}"),
            ("Distinct Items", int(rrow["distinct_items_served"])),
            ("Top Item", rrow["top_ordered_item"]),
        ]
        for k, v in rrows:
            st.markdown(profile_row(k, v), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with pr2:
        hs = rrow["restaurant_health_score"]
        is_u = hs < 0.6
        hc = "#F87171" if is_u else "#34D399"
        hl = "Unhealthy" if is_u else "Healthy"
        hb = "red" if is_u else "green"

        st.markdown(f"""
        <div class='pred-card'>
            <div class='pred-label'>Health Score</div>
            <div class='pred-score' style='color:{hc};'>{hs:.3f}</div>
            <div style='margin:0.6rem 0;'>{badge(hl, hb)}</div>
            <div class='prog-bg'><div class='prog-fill' style='width:{hs*100:.0f}%;background:{hc};'></div></div>
            <div style='font-size:0.78rem;color:#64748B;margin-top:0.5rem;'>
                Threshold: 0.6 · {'Below threshold — review required' if is_u else 'Above threshold — performing well'}
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        imps = rest_model.feature_importances_
        fi = sorted(zip(rest_feats, imps), key=lambda x: x[1], reverse=True)[:6]
        max_i = max(f[1] for f in fi) or 1

        st.markdown("<div class='input-card'><div class='input-card-title'>Top Feature Importances</div>", unsafe_allow_html=True)
        for fn, fv in fi:
            st.markdown(feat_bar(fn, fv, max_i), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        insight(f"{'This restaurant is flagged as unhealthy. High cancellation rate and poor delivery times are the primary contributors. Immediate partnership review is recommended.' if is_u else 'This restaurant is performing well above the health threshold. Strong operational metrics and low cancellation rate indicate a reliable platform partner.'}")