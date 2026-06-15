import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import joblib

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIG & DESIGN SYSTEM (SWISS MODERNISM)
# ==========================================
st.set_page_config(
    page_title="EV Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global settings */
    .stApp {
        background-color: #f4f4f0;
        color: #0f0f0f;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #0f0f0f !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        text-transform: uppercase;
    }

    p, span, div, label {
        font-family: 'Space Grotesk', sans-serif;
        color: #0f0f0f;
    }

    /* 1. Hide default Streamlit Deploy Button and Main Menu */
    .stAppDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    
    /* Ensure the header is transparent to avoid black blocks */
    header { background-color: transparent !important; }

    /* 2. Fix the "Invisible" Maximize Button 
       Streamlit paints it white by default in dark mode. 
       We must force the SVG icon to be black (#0f0f0f) so it shows up on the cream background! */
    [data-testid="collapsedControl"] svg,
    [data-testid="collapsedSidebarCollapsed"] svg,
    header svg {
        fill: #0f0f0f !important;
        color: #0f0f0f !important;
    }

    /* Sidebar Brutalism */
    [data-testid="stSidebar"] {
        background-color: #e5e5e0 !important;
        border-right: 2px solid #0f0f0f !important;
    }
    
    [data-testid="stSidebar"] h1 {
        border-bottom: 2px solid #0f0f0f;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Custom Streamlit Button */
    .stButton > button {
        background-color: #0f0f0f !important;
        color: #ffffff !important;
        border-radius: 0px !important;
        border: 2px solid #0f0f0f !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        width: 100%;
        padding: 12px !important;
        transition: all 0.2s cubic-bezier(0.25, 1, 0.5, 1);
        box-shadow: 4px 4px 0px 0px rgba(15,15,15,1) !important;
    }
    .stButton > button * {
        color: #ffffff !important;
    }
    .stButton > button:hover {
        background-color: #ff3300 !important;
        border-color: #ff3300 !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px 0px rgba(255,51,0,0.5) !important;
        color: #ffffff !important;
    }
    .stButton > button:active {
        transform: translate(2px, 2px);
        box-shadow: 0px 0px 0px 0px rgba(15,15,15,1) !important;
    }

    /* Custom Slider */
    .stSlider > div[data-baseweb="slider"] {
        padding-top: 10px;
    }
    div[role="slider"] {
        background-color: #ff3300 !important;
        border: 2px solid #0f0f0f !important;
    }
    div[data-baseweb="slider"] > div > div > div {
        background-color: #0f0f0f !important;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'JetBrains Mono', monospace;
        background-color: #f4f4f0;
        border: 2px solid #0f0f0f;
        margin-top: 20px;
    }
    th {
        background-color: #0f0f0f;
        color: #f4f4f0;
        font-weight: 700;
        text-transform: uppercase;
        padding: 12px;
        border: 1px solid #0f0f0f;
        font-size: 0.85rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    td {
        padding: 12px;
        border: 1px solid #0f0f0f;
        color: #0f0f0f;
        font-size: 0.95rem;
    }
    tr:hover {
        background-color: #e5e5e0;
    }

    /* Brutalist Summary Cards for bottom section */
    .brutal-card {
        border: 2px solid #0f0f0f;
        background-color: #ffffff;
        padding: 20px;
        box-shadow: 4px 4px 0px 0px rgba(15,15,15,1);
    }
    .brutal-card-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
        color: #555;
    }
    .brutal-card-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f0f0f;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_data
def load_historical_data():
    df = pd.read_csv('ev_population.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df

@st.cache_resource
def load_model_and_scaler():
    model = load_model('ev_model.h5', compile=False)
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_metrics():
    try:
        with open('metrics.json', 'r') as f:
            return json.load(f)
    except:
        return {"mape": 0.0, "mae": 0.0, "rmse": 0.0}

def render_hero(mape, mae, rmse, latest_count):
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                background-color: transparent;
                font-family: 'Space Grotesk', sans-serif;
                color: #0f0f0f;
                padding: 10px;
                padding-top: 15px;
            }}
            .header-container {{
                margin-bottom: 30px;
                opacity: 0;
                transform: translateY(20px);
            }}
            h1 {{
                font-size: 2.5rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: -0.05em;
                margin-bottom: 5px;
                color: #0f0f0f;
            }}
            .subtitle {{
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #555;
                border-bottom: 2px solid #0f0f0f;
                padding-bottom: 15px;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
            }}
            @media (max-width: 768px) {{
                .grid {{ grid-template-columns: repeat(2, 1fr); }}
                h1 {{ font-size: 2rem; }}
            }}
            .card {{
                border: 2px solid #0f0f0f;
                box-shadow: 4px 4px 0px 0px rgba(15,15,15,1);
                background-color: #ffffff;
                padding: 15px;
                opacity: 0;
                position: relative;
                overflow: hidden;
            }}
            .card-dark {{ background-color: #0f0f0f; color: #f4f4f0; }}
            .card-label {{
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #777;
                margin-bottom: 10px;
            }}
            .card-dark .card-label {{ color: #aaa; }}
            .metric-val {{
                font-family: 'JetBrains Mono', monospace;
                font-size: 1.8rem;
                font-weight: 700;
            }}
            .text-blue {{ color: #0033cc; }}
            .text-orange {{ color: #ff3300; }}
            .accent-corner {{
                position: absolute;
                top: 0;
                right: 0;
                width: 30px;
                height: 30px;
                background-color: #0033cc;
                border-left: 2px solid #0f0f0f;
                border-bottom: 2px solid #0f0f0f;
            }}
            .flex-row {{ display: flex; align-items: flex-end; gap: 5px; }}
        </style>
    </head>
    <body>
        <div class="header-container">
            <h1>EV Forecasting</h1>
            <p class="subtitle">Washington State // Predictive Intelligence</p>
        </div>

        <div class="grid">
            <div class="card">
                <div class="accent-corner"></div>
                <p class="card-label">MAPE (Error)</p>
                <div class="flex-row">
                    <span class="metric-val text-blue" id="mape-val">0</span>
                    <span class="metric-val text-blue" style="font-size: 1.2rem;">%</span>
                </div>
            </div>
            
            <div class="card">
                <p class="card-label">Mean Abs Err (MAE)</p>
                <div class="metric-val text-orange" id="mae-val">0</div>
            </div>
            
            <div class="card">
                <p class="card-label">Root Mean Sq (RMSE)</p>
                <div class="metric-val text-orange" id="rmse-val">0</div>
            </div>
            
            <div class="card card-dark">
                <p class="card-label">Latest EV Count</p>
                <div class="metric-val text-orange" id="latest-val">0</div>
            </div>
        </div>

        <script>
            // Entrance animations
            anime.timeline({{ easing: 'easeOutExpo' }})
            .add({{
                targets: '.header-container',
                translateY: [20, 0],
                opacity: [0, 1],
                duration: 1000
            }})
            .add({{
                targets: '.card',
                translateY: [30, 0],
                opacity: [0, 1],
                delay: anime.stagger(100),
                duration: 800
            }}, '-=600');

            // Number counters
            const formatNumber = (num) => num.toLocaleString('en-US');
            
            const countUp = (targets, maxVal, isFloat=false) => {{
                let obj = {{ val: 0 }};
                anime({{
                    targets: obj,
                    val: maxVal,
                    round: isFloat ? 100 : 1,
                    duration: 2000,
                    easing: 'easeOutExpo',
                    update: function() {{
                        if(isFloat) {{
                            document.querySelector(targets).innerHTML = (obj.val).toFixed(2);
                        }} else {{
                            document.querySelector(targets).innerHTML = formatNumber(Math.round(obj.val));
                        }}
                    }}
                }});
            }};

            // Small delay to ensure browser render thread is free
            setTimeout(() => {{
                countUp('#mape-val', {mape}, true);
                countUp('#mae-val', {mae}, false);
                countUp('#rmse-val', {rmse}, false);
                countUp('#latest-val', {latest_count}, false);
            }}, 100);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=280)


# ==========================================
# 3. MAIN APP LOGIC
# ==========================================
df = load_historical_data()
model, scaler = load_model_and_scaler()
metrics = load_metrics()

latest_date = df['Date'].max()
latest_count = int(df['Electric Vehicle (EV) Total'].iloc[-1])

# Render Custom Hero Section via iframe (Tailwind + AnimeJS)
render_hero(metrics['mape'], metrics['mae'], metrics['rmse'], latest_count)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("<h1>Config</h1>", unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight:700; font-size:0.85rem; text-transform:uppercase;'>Prediction Horizon</p>", unsafe_allow_html=True)
    months_to_predict = st.slider("Months", min_value=1, max_value=24, value=6, label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("Run Forecast")
    
    st.markdown("<hr style='border:1px solid #0f0f0f; margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-size:1rem;'>Model Specs</h3>", unsafe_allow_html=True)
    st.markdown("""
    <ul style='font-family: "JetBrains Mono", monospace; font-size: 0.8rem; padding-left: 15px;'>
        <li>Arch: 2-Layer LSTM</li>
        <li>Lookback: 12 months</li>
        <li>Optimizer: Adam</li>
        <li>Loss: MSE</li>
    </ul>
    """, unsafe_allow_html=True)

# ----------------- PLOT STYLING -----------------
def create_brutalist_chart(df_hist, df_pred=None):
    fig = go.Figure()

    # Historical Data Line
    fig.add_trace(go.Scatter(
        x=df_hist['Date'], 
        y=df_hist['Electric Vehicle (EV) Total'],
        mode='lines',
        name='Historical',
        line=dict(color='#0f0f0f', width=3),
        hovertemplate='%{x|%b %Y}<br><b>%{y:,.0f} EV</b><extra></extra>'
    ))

    # Prediction Data
    if df_pred is not None:
        # Connect the last historical point to the first prediction point
        connect_x = [df_hist['Date'].iloc[-1], df_pred['Date'].iloc[0]]
        connect_y = [df_hist['Electric Vehicle (EV) Total'].iloc[-1], df_pred['Predicted_EV'].iloc[0]]
        
        fig.add_trace(go.Scatter(
            x=connect_x, y=connect_y,
            mode='lines', showlegend=False,
            line=dict(color='#ff3300', width=3, dash='dot')
        ))

        # Prediction Line
        fig.add_trace(go.Scatter(
            x=df_pred['Date'],
            y=df_pred['Predicted_EV'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#ff3300', width=3, dash='dot'),
            marker=dict(size=8, color='#ff3300', symbol='square', line=dict(color='#0f0f0f', width=1)),
            hovertemplate='%{x|%b %Y}<br><b>%{y:,.0f} EV</b> (Proj.)<extra></extra>'
        ))

        # Confidence Band (Simplified visual representation)
        upper_bound = df_pred['Predicted_EV'] * 1.05
        lower_bound = df_pred['Predicted_EV'] * 0.95
        
        fig.add_trace(go.Scatter(
            x=pd.concat([df_pred['Date'], df_pred['Date'][::-1]]),
            y=pd.concat([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(255, 51, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='95% Confidence'
        ))

    # Layout styling for Swiss Modernism / Brutalism
    fig.update_layout(
        paper_bgcolor='#f4f4f0',
        plot_bgcolor='#f4f4f0',
        font=dict(color='#0f0f0f', family='JetBrains Mono'),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(family='Space Grotesk', size=12, color='#0f0f0f')
        ),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='#e5e5e0',
            zeroline=True, zerolinewidth=2, zerolinecolor='#0f0f0f',
            showline=True, linewidth=2, linecolor='#0f0f0f',
            tickfont=dict(family='JetBrains Mono', color='#0f0f0f')
        ),
        yaxis=dict(
            showgrid=True, gridwidth=1, gridcolor='#e5e5e0',
            zeroline=True, zerolinewidth=2, zerolinecolor='#0f0f0f',
            showline=True, linewidth=2, linecolor='#0f0f0f',
            tickfont=dict(family='JetBrains Mono', color='#0f0f0f')
        ),
        hoverlabel=dict(
            bgcolor="#0f0f0f",
            font_size=14,
            font_family="JetBrains Mono",
            font_color="#f4f4f0",
            bordercolor="#ff3300"
        )
    )
    return fig

# ----------------- MAIN VIEW -----------------
# Initial chart (History only)
if not run_btn:
    st.plotly_chart(create_brutalist_chart(df), use_container_width=True)
    st.markdown("<p style='text-align:right; font-family:\"JetBrains Mono\"; font-size:0.8rem;'>Awaiting Forecast Execution...</p>", unsafe_allow_html=True)

# Forecast Execution
if run_btn:
    with st.spinner("Compiling Neural Network Forecast..."):
        # Prepare input data (last 12 months)
        lookback = 12
        last_data = df['Electric Vehicle (EV) Total'].values[-lookback:]
        last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
        
        current_sequence = last_data_scaled.reshape(1, lookback, 1)
        predictions_scaled = []
        
        # Iterative prediction
        for _ in range(months_to_predict):
            pred = model.predict(current_sequence, verbose=0)
            predictions_scaled.append(pred[0, 0])
            # Update sequence: remove oldest, add new prediction
            current_sequence = np.append(current_sequence[:, 1:, :], [[[pred[0, 0]]]], axis=1)
            
        # Inverse transform
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        
        # Create Dates
        future_dates = [latest_date + pd.DateOffset(months=i) for i in range(1, months_to_predict + 1)]
        
        df_pred = pd.DataFrame({
            'Date': future_dates,
            'Predicted_EV': predictions
        })
        
        # Draw Chart
        st.plotly_chart(create_brutalist_chart(df, df_pred), use_container_width=True)
        
        st.markdown("<h3>Forecast Data</h3>", unsafe_allow_html=True)
        
        # Generate HTML Table
        table_rows = ""
        prev_val = latest_count
        
        for idx, row in df_pred.iterrows():
            date = row['Date']
            value = row['Predicted_EV']
            change = value - prev_val
            pct = (change / prev_val) * 100
            prev_val = value
            
            arrow = "▲" if change >= 0 else "▼"
            color = "#0033cc" if change >= 0 else "#ff3300"
            bg = "#ffffff" if idx % 2 == 0 else "#f4f4f0"

            table_rows += f"""<tr style="background-color:{bg};">
<td>{date.strftime('%B %Y')}</td>
<td style="font-weight:700;">{value:,.0f}</td>
<td style="color:{color}; font-weight:700;">
    {arrow} {abs(change):,.0f} ({pct:+.1f}%)
</td>
</tr>"""

        st.markdown(f"""<table>
<thead>
    <tr>
        <th>Month</th>
        <th>Predicted Total</th>
        <th>Monthly Change</th>
    </tr>
</thead>
<tbody>
    {table_rows}
</tbody>
</table>""", unsafe_allow_html=True)

        # Summary Cards
        total_growth = df_pred['Predicted_EV'].iloc[-1] - latest_count
        growth_pct = (total_growth / latest_count) * 100
        avg_monthly = total_growth / months_to_predict
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="brutal-card">
                <div class="brutal-card-label">Proj. Growth</div>
                <div class="brutal-card-val text-[#0033cc]">{total_growth:+,.0f}</div>
                <div style="font-size:0.8rem; margin-top:5px;">({growth_pct:+.1f}%)</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="brutal-card">
                <div class="brutal-card-label">Final Target</div>
                <div class="brutal-card-val">{df_pred['Predicted_EV'].iloc[-1]:,.0f}</div>
                <div style="font-size:0.8rem; margin-top:5px;">EVs by {df_pred['Date'].iloc[-1].strftime('%b %Y')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="brutal-card" style="background-color:#0f0f0f;">
                <div class="brutal-card-label" style="color:#e5e5e0;">Avg Monthly Inc.</div>
                <div class="brutal-card-val" style="color:#f4f4f0;">{avg_monthly:+,.0f}</div>
                <div style="font-size:0.8rem; margin-top:5px; color:#e5e5e0;">vehicles/month</div>
            </div>
            """, unsafe_allow_html=True)
