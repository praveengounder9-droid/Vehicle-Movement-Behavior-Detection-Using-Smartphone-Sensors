"""
Phone Motion and Driving Behavior Detection Dashboard
Interactive Streamlit Web Application
Preserving exact methodology from Project.ipynb
(L.D. College of Engineering - AIML Department)
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core_analysis import (
    process_sensor_datasets,
    explain_single_observation,
    answer_user_query,
    generate_markdown_report,
    REQUIRED_COLUMNS,
)

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Phone Motion & Driving Behavior Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphic Dark UI Theme
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, kbd, samp, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Main container styling */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* Hero Header Card */
.hero-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(12px);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.5;
    margin-bottom: 0.8rem;
}

.academic-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.85rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(129, 140, 248, 0.3);
    border-radius: 9999px;
    color: #c7d2fe;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

/* Metric KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.kpi-card {
    background: rgba(30, 41, 59, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.25);
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.4);
}

.kpi-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.3rem;
}

.kpi-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.01em;
}

.kpi-hint {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
}

/* Classification Badges */
.badge-normal {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-turning {
    background: rgba(245, 158, 11, 0.2);
    border: 1px solid #f59e0b;
    color: #fbbf24;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

.badge-braking {
    background: rgba(244, 63, 94, 0.2);
    border: 1px solid #f43f5e;
    color: #fb7185;
    padding: 0.3rem 0.75rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

/* Explanation Box */
.explanation-box {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 1.2rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

.explanation-title {
    color: #818cf8;
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.5rem;
}

/* Question Suggestion Buttons */
.query-chip {
    cursor: pointer;
    background: rgba(51, 65, 85, 0.5);
    border: 1px solid rgba(100, 116, 139, 0.4);
    border-radius: 8px;
    padding: 0.45rem 0.8rem;
    color: #cbd5e1;
    font-size: 0.82rem;
    margin: 0.25rem;
    transition: all 0.2s ease;
    display: inline-block;
}

.query-chip:hover {
    background: rgba(99, 102, 241, 0.3);
    border-color: #818cf8;
    color: #ffffff;
}

/* Tab styling adjustments */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 16px;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Color Scheme Mapping for Events
EVENT_COLORS = {
    "Normal Movement": "#10b981",          # Emerald
    "Sudden Acceleration/Braking": "#f43f5e", # Rose/Red
    "Turning": "#f59e0b",                  # Amber
}

# ---------------------------------------------------------
# Sidebar - Data Loading & Filtering
# ---------------------------------------------------------
with st.sidebar:
    st.image(
        "https://images.unsplash.com/photo-1549399542-7e3f8b79c341?auto=format&fit=crop&w=400&q=80",
        caption="Vehicle Telematics & Smartphone Sensors",
        use_container_width=True,
    )
    st.markdown("### 🎛️ Control Center")

    # Academic metadata
    with st.expander("🎓 Project & Author Information", expanded=False):
        st.markdown(
            """
            **Institution**: L.D. College of Engineering, Ahmedabad  
            **Program**: B.E. Artificial Intelligence & Machine Learning  
            **Student**: Gounder Praveen Murugesan  
            **Enrollment**: 240280152025  
            **Course**: Python Programming (BE05000241)  
            **Core Concept**: Smartphone IMU Sensor Fusion & Statistical Maneuver Detection
            """
        )

    st.markdown("---")
    st.markdown("#### 📂 Dataset Source")

    data_source = st.radio(
        "Choose Data Source:",
        ["Default Project Files (Workspace)", "Upload Custom Sensor Files"],
        index=0,
        help="Select whether to use the pre-bundled Accelerometer.csv and Gyroscope.csv or upload new ones."
    )

    acc_file = None
    gyro_file = None

    if data_source == "Default Project Files (Workspace)":
        default_acc = "Accelerometer.csv"
        default_gyro = "Gyroscope.csv"
        if os.path.exists(default_acc) and os.path.exists(default_gyro):
            acc_file = default_acc
            gyro_file = default_gyro
            st.success(" Loaded pre-bundled project datasets (`Accelerometer.csv` & `Gyroscope.csv`).")
        else:
            st.error("Default files not found in workspace directory.")
    else:
        st.markdown("**1. Accelerometer File** (CSV or Excel)")
        acc_file = st.file_uploader(
            "Upload Accelerometer",
            type=["csv", "xlsx", "xls"],
            key="acc_uploader",
            help="Must contain columns: Milliseconds, X, Y, Z (Timestamp optional)"
        )

        st.markdown("**2. Gyroscope File** (CSV or Excel)")
        gyro_file = st.file_uploader(
            "Upload Gyroscope",
            type=["csv", "xlsx", "xls"],
            key="gyro_uploader",
            help="Must contain columns: Milliseconds, X, Y, Z (Timestamp optional)"
        )

    st.markdown("---")

# ---------------------------------------------------------
# Data Processing Pipeline Execution
# ---------------------------------------------------------
if acc_file is None or gyro_file is None:
    st.info("👋 Welcome! Please ensure both Accelerometer and Gyroscope datasets are selected in the sidebar to begin analysis.")
    st.stop()

@st.cache_data(show_spinner=False)
def run_cached_pipeline(acc_input, gyro_input):
    return process_sensor_datasets(acc_input, gyro_input)

try:
    with st.spinner("Processing sensor readings using exact Project.ipynb pipeline..."):
        pipeline_result = run_cached_pipeline(acc_file, gyro_file)
except Exception as e:
    st.error(f"❌ Error processing sensor datasets: {str(e)}")
    st.markdown("Please verify that your files contain `Milliseconds`, `X`, `Y`, and `Z` columns.")
    st.stop()

# Extract Results
full_df = pipeline_result["dataframe"]
stats = pipeline_result["stats"]
acc_threshold = stats["acc_threshold"]
gyro_threshold = stats["gyro_threshold"]

# ---------------------------------------------------------
# Sidebar - Interactive Filters
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("#### 🔍 Filter Observations")

    # Event Type Filter
    all_events = ["Normal Movement", "Sudden Acceleration/Braking", "Turning"]
    selected_events = st.multiselect(
        "Event Categories:",
        options=all_events,
        default=all_events,
        help="Filter dashboard visuals and table by detected event category."
    )

    # Milliseconds / Time Range Filter
    min_ms = int(full_df["Milliseconds"].min())
    max_ms = int(full_df["Milliseconds"].max())

    ms_range = st.slider(
        "Time Range (Milliseconds):",
        min_value=min_ms,
        max_value=max_ms,
        value=(min_ms, max_ms),
        step=max(1, (max_ms - min_ms) // 100),
        help="Select a specific time window for detailed inspection."
    )

    # Feature Threshold Filter option
    filter_anomalies_only = st.checkbox(
        "Show Anomaly Maneuvers Only",
        value=False,
        help="Quickly isolate Sudden Acceleration/Braking and Turning events."
    )

    if st.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state["acc_uploader"] = None
        st.session_state["gyro_uploader"] = None
        st.rerun()

# Apply Filters to DataFrame
filtered_df = full_df.copy()
if filter_anomalies_only:
    filtered_df = filtered_df[filtered_df["Event"] != "Normal Movement"]
else:
    if selected_events:
        filtered_df = filtered_df[filtered_df["Event"].isin(selected_events)]
    else:
        filtered_df = filtered_df.iloc[0:0]

filtered_df = filtered_df[
    (filtered_df["Milliseconds"] >= ms_range[0]) &
    (filtered_df["Milliseconds"] <= ms_range[1])
]

# ---------------------------------------------------------
# Top Hero Banner & Academic Context
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
            <div>
                <div class="hero-title">Phone Motion & Driving Behavior Detection</div>
                <div class="hero-subtitle">
                    Automated vehicle motion classification via smartphone Inertial Measurement Unit (IMU) sensor fusion.
                </div>
            </div>
            <div>
                <span class="academic-badge">🏛️ L.D. College of Engineering (AIML)</span>
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 0.4rem; margin-bottom: 0;">
            <b>Sensor Mechanics</b>: <b>Accelerometer</b> captures proper linear acceleration (including gravitational baseline $1g \\approx 9.8$ m/s²) to identify sudden braking and acceleration. 
            <b>Gyroscope</b> captures rotational angular velocity in rad/s to identify turns, vehicle yaw changes, and lateral maneuvers.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Section 1: KPI Summary Metrics
# ---------------------------------------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Raw Inputs</div>
            <div class="kpi-value">{stats['raw_acc_rows'] + stats['raw_gyro_rows']:,}</div>
            <div class="kpi-hint">Acc: {stats['raw_acc_rows']:,} | Gyro: {stats['raw_gyro_rows']:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Merged Records</div>
            <div class="kpi-value">{stats['merged_rows']:,}</div>
            <div class="kpi-hint">Duplicates purged: {stats['acc_duplicates_removed'] + stats['gyro_duplicates_removed']:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Maneuvers Detected</div>
            <div class="kpi-value" style="color: #fb7185;">{stats['detected_anomalies']:,}</div>
            <div class="kpi-hint">Brake/Acc: {stats['event_counts']['Sudden Acceleration/Braking']} | Turn: {stats['event_counts']['Turning']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Predominant Class</div>
            <div class="kpi-value" style="color: #34d399; font-size: 1.4rem;">{stats['most_common_class']}</div>
            <div class="kpi-hint">{stats['event_percentages'][stats['most_common_class']]:.1f}% of entire journey</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Acc Threshold</div>
            <div class="kpi-value" style="color: #38bdf8;">{stats['acc_threshold']:.3f}</div>
            <div class="kpi-hint">m/s² (Mean + 2σ: {stats['acc_change_mean']:.2f} + 2x{stats['acc_change_std']:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Gyro Threshold</div>
            <div class="kpi-value" style="color: #fbbf24;">{stats['gyro_threshold']:.3f}</div>
            <div class="kpi-hint">rad/s (Mean + 2σ: {stats['gyro_mag_mean']:.2f} + 2x{stats['gyro_mag_std']:.2f})</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Organization
# ---------------------------------------------------------
tab_charts, tab_explorer, tab_inspect, tab_insights, tab_qa, tab_export = st.tabs([
    "📊 Interactive Visualizations",
    "🧭 Sensor Data Explorer",
    "🎯 Observation Inspector",
    "💡 What Do Results Mean?",
    "💬 Ask Questions About Data",
    "📥 Export & Technical Report"
])

# ---------------------------------------------------------
# TAB 1: INTERACTIVE VISUALIZATIONS
# ---------------------------------------------------------
with tab_charts:
    st.markdown("### 📈 Interactive Sensor & Event Visualizations")
    st.caption("Visualizing synchronized sensor metrics and the statistical decision boundaries established in Project.ipynb.")

    chart_view = st.radio(
        "Select Visualization Perspective:",
        [
            "⚡ Acceleration Changes & Threshold Over Time",
            "🌀 Gyroscope Magnitude & Rotational Threshold Over Time",
            "🛣️ Driving Behavior Event Timeline",
            "🥧 Event Category Distribution",
            "🌌 3D Sensor Trajectory Space"
        ],
        horizontal=True
    )

    if filtered_df.empty:
        st.warning("⚠️ No data points match the current filter selection. Please expand your filter criteria in the sidebar.")
    else:
        # 1. Acceleration Change with Threshold
        if "Acceleration Changes" in chart_view:
            fig_acc = go.Figure()
            
            # Line of Acceleration Change
            fig_acc.add_trace(go.Scatter(
                x=filtered_df["Milliseconds"],
                y=filtered_df["Acceleration_Change"],
                mode="lines",
                name="Δ Acceleration (|diff|)",
                line=dict(color="#38bdf8", width=1.5),
                hovertemplate="<b>Time:</b> %{x} ms<br><b>Δ Acc:</b> %{y:.3f} m/s²<extra></extra>"
            ))

            # Threshold Reference Line
            fig_acc.add_hline(
                y=acc_threshold,
                line_dash="dash",
                line_color="#f43f5e",
                line_width=2.5,
                annotation_text=f"Calculated Threshold (μ + 2σ = {acc_threshold:.3f} m/s²)",
                annotation_position="top left",
                annotation_font=dict(color="#f43f5e", size=12)
            )

            # Highlight Sudden Acc/Braking points
            sudden_points = filtered_df[filtered_df["Event"] == "Sudden Acceleration/Braking"]
            if not sudden_points.empty:
                fig_acc.add_trace(go.Scatter(
                    x=sudden_points["Milliseconds"],
                    y=sudden_points["Acceleration_Change"],
                    mode="markers",
                    name="Detected Sudden Acc/Brake",
                    marker=dict(color="#f43f5e", size=8, symbol="diamond"),
                    hovertemplate="<b>Event:</b> Sudden Acceleration/Braking<br><b>Time:</b> %{x} ms<br><b>Δ Acc:</b> %{y:.3f} m/s²<extra></extra>"
                ))

            fig_acc.update_layout(
                title="Acceleration Changes Over Time (with Statistical Decision Threshold)",
                xaxis_title="Time Elapsed (Milliseconds)",
                yaxis_title="Acceleration Change Magnitude (m/s²)",
                template="plotly_dark",
                height=520,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        # 2. Gyroscope Magnitude with Threshold
        elif "Gyroscope Magnitude" in chart_view:
            fig_gyro = go.Figure()

            fig_gyro.add_trace(go.Scatter(
                x=filtered_df["Milliseconds"],
                y=filtered_df["Gyroscope_Magnitude"],
                mode="lines",
                name="Gyroscope Magnitude",
                line=dict(color="#fbbf24", width=1.5),
                hovertemplate="<b>Time:</b> %{x} ms<br><b>Gyro Mag:</b> %{y:.3f} rad/s<extra></extra>"
            ))

            fig_gyro.add_hline(
                y=gyro_threshold,
                line_dash="dash",
                line_color="#ec4899",
                line_width=2.5,
                annotation_text=f"Calculated Threshold (μ + 2σ = {gyro_threshold:.3f} rad/s)",
                annotation_position="top left",
                annotation_font=dict(color="#ec4899", size=12)
            )

            turning_points = filtered_df[filtered_df["Event"] == "Turning"]
            if not turning_points.empty:
                fig_gyro.add_trace(go.Scatter(
                    x=turning_points["Milliseconds"],
                    y=turning_points["Gyroscope_Magnitude"],
                    mode="markers",
                    name="Detected Turning Maneuver",
                    marker=dict(color="#f59e0b", size=8, symbol="circle"),
                    hovertemplate="<b>Event:</b> Turning<br><b>Time:</b> %{x} ms<br><b>Gyro Mag:</b> %{y:.3f} rad/s<extra></extra>"
                ))

            fig_gyro.update_layout(
                title="Gyroscope Rotational Velocity Over Time (with Turning Decision Threshold)",
                xaxis_title="Time Elapsed (Milliseconds)",
                yaxis_title="Angular Velocity Magnitude (rad/s)",
                template="plotly_dark",
                height=520,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_gyro, use_container_width=True)

        # 3. Driving Behavior Event Timeline
        elif "Event Timeline" in chart_view:
            fig_timeline = px.scatter(
                filtered_df,
                x="Milliseconds",
                y="Acceleration_Magnitude",
                color="Event",
                color_discrete_map=EVENT_COLORS,
                size=np.where(filtered_df["Event"] == "Normal Movement", 4, 10),
                hover_data=["Milliseconds", "Acceleration_Change", "Gyroscope_Magnitude"],
                title="Driving Behavior Classification Timeline (Acceleration Magnitude by Event Category)",
                labels={
                    "Milliseconds": "Time Elapsed (ms)",
                    "Acceleration_Magnitude": "Total Acceleration Vector Magnitude (m/s²)",
                    "Event": "Classified Event"
                },
                template="plotly_dark",
                height=520
            )
            fig_timeline.add_hline(
                y=9.81,
                line_dash="dot",
                line_color="#94a3b8",
                annotation_text="Earth Gravity Reference (1g ≈ 9.81 m/s²)",
                annotation_position="bottom right"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # 4. Distribution Charts
        elif "Distribution" in chart_view:
            col_d1, col_d2 = st.columns(2)

            counts_series = filtered_df["Event"].value_counts()
            dist_df = pd.DataFrame({
                "Event": counts_series.index,
                "Count": counts_series.values,
                "Percentage": (counts_series.values / len(filtered_df) * 100).round(2)
            })

            with col_d1:
                fig_donut = px.pie(
                    dist_df,
                    names="Event",
                    values="Count",
                    color="Event",
                    color_discrete_map=EVENT_COLORS,
                    hole=0.55,
                    title=f"Event Category Proportions (Filtered: {len(filtered_df):,} records)",
                    template="plotly_dark",
                )
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_donut, use_container_width=True)

            with col_d2:
                fig_bar = px.bar(
                    dist_df,
                    x="Event",
                    y="Count",
                    color="Event",
                    color_discrete_map=EVENT_COLORS,
                    text="Count",
                    title="Absolute Observation Counts per Category",
                    template="plotly_dark",
                )
                fig_bar.update_traces(texttemplate='%{text:,}', textposition='outside')
                fig_bar.update_layout(yaxis_title="Number of Observations", showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        # 5. 3D Sensor Trajectory Space
        elif "3D Sensor Trajectory Space" in chart_view:
            fig_3d = px.scatter_3d(
                filtered_df,
                x="Acc_X",
                y="Acc_Y",
                z="Acc_Z",
                color="Event",
                color_discrete_map=EVENT_COLORS,
                opacity=0.7,
                size=np.where(filtered_df["Event"] == "Normal Movement", 2, 6),
                title="3D Accelerometer Orientation Dynamics (Acc_X vs Acc_Y vs Acc_Z)",
                labels={"Acc_X": "Acc X (m/s²)", "Acc_Y": "Acc Y (m/s²)", "Acc_Z": "Acc Z (m/s²)"},
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: SENSOR DATA EXPLORER
# ---------------------------------------------------------
with tab_explorer:
    st.markdown("### 🧭 Sensor Data Explorer")
    st.caption("Inspect, sort, filter, and review synchronized sensor readings and derived motion metrics.")

    e1, e2 = st.columns([3, 1])
    with e1:
        st.write(f"Showing **{len(filtered_df):,}** of **{len(full_df):,}** observations based on active sidebar filters.")
    with e2:
        search_kw = st.text_input("🔍 Quick Search by Milliseconds:", placeholder="e.g. 1042")

    display_df = filtered_df.copy()
    if search_kw.strip().isdigit():
        target_ms = int(search_kw.strip())
        display_df = display_df[display_df["Milliseconds"] == target_ms]

    # Selectable columns
    all_display_cols = list(display_df.columns)
    default_cols = [
        col for col in [
            "Milliseconds", "Timestamp", "Acc_X", "Acc_Y", "Acc_Z",
            "Gyro_X", "Gyro_Y", "Gyro_Z", "Acceleration_Magnitude",
            "Gyroscope_Magnitude", "Acceleration_Change", "Event"
        ] if col in all_display_cols
    ]

    selected_columns = st.multiselect(
        "Toggle Visible Columns:",
        options=all_display_cols,
        default=default_cols
    )

    if not display_df.empty and selected_columns:
        # Format floating numbers nicely
        format_dict = {
            col: "{:.4f}" for col in selected_columns
            if col in ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z",
                       "Acceleration_Magnitude", "Gyroscope_Magnitude",
                       "Acceleration_Change", "Gyroscope_Change"]
        }
        st.dataframe(
            display_df[selected_columns].style.format(format_dict, na_rep="N/A"),
            use_container_width=True,
            height=420
        )
    else:
        st.info("No matching records to display.")

# ---------------------------------------------------------
# TAB 3: OBSERVATION INSPECTOR
# ---------------------------------------------------------
with tab_inspect:
    st.markdown("### 🎯 Single Observation Deep-Dive & Rule Explainer")
    st.caption("Select any specific row to inspect its raw readings, calculated metrics, and the step-by-step decision rule that classified it.")

    if full_df.empty:
        st.warning("No data available.")
    else:
        # Observation selection mechanism
        insp_col1, insp_col2 = st.columns([1, 2])
        with insp_col1:
            sel_mode = st.radio("Selection Method:", ["Select by Row Index", "Select by Milliseconds Timestamp"], horizontal=True)
            
            if sel_mode == "Select by Row Index":
                chosen_idx = st.number_input(
                    f"Row Index (0 to {len(full_df)-1}):",
                    min_value=0,
                    max_value=len(full_df)-1,
                    value=18, # Row 18 is Turning in sample dataset
                    step=1
                )
            else:
                available_ms = full_df["Milliseconds"].tolist()
                chosen_ms = st.selectbox("Select Millisecond Value:", available_ms, index=min(18, len(available_ms)-1))
                chosen_idx = full_df[full_df["Milliseconds"] == chosen_ms].index[0]

        target_row = full_df.iloc[chosen_idx]
        explanation = explain_single_observation(target_row, acc_threshold, gyro_threshold)

        with insp_col2:
            event_name = target_row["Event"]
            badge_class = "badge-normal"
            if event_name == "Turning":
                badge_class = "badge-turning"
            elif event_name == "Sudden Acceleration/Braking":
                badge_class = "badge-braking"

            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                            Observation #{chosen_idx} (Time: {target_row['Milliseconds']} ms)
                        </span>
                        <span class="{badge_class}">{event_name}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.4rem;">
                        {explanation['reasoning_summary']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### 🔬 Detailed Metrics Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric(
                label="Acceleration Magnitude",
                value=f"{target_row['Acceleration_Magnitude']:.3f} m/s²",
                delta=f"{(target_row['Acceleration_Magnitude'] - 9.81):+.2f} vs 1g"
            )
            st.caption(f"Acc X: {target_row['Acc_X']:.2f} | Y: {target_row['Acc_Y']:.2f} | Z: {target_row['Acc_Z']:.2f}")

        with m2:
            acc_ch = target_row['Acceleration_Change']
            acc_ch_str = "N/A" if np.isnan(acc_ch) else f"{acc_ch:.3f} m/s²"
            delta_str = "N/A" if np.isnan(acc_ch) else f"{(acc_ch - acc_threshold):+.3f} vs threshold"
            st.metric(
                label="Acceleration Change (Δ Acc)",
                value=acc_ch_str,
                delta=delta_str,
                delta_color="inverse"
            )
            st.caption(f"Threshold: **{acc_threshold:.3f} m/s²**")

        with m3:
            gyro_mag = target_row['Gyroscope_Magnitude']
            st.metric(
                label="Gyroscope Magnitude",
                value=f"{gyro_mag:.3f} rad/s",
                delta=f"{(gyro_mag - gyro_threshold):+.3f} vs threshold",
                delta_color="inverse"
            )
            st.caption(f"Gyro X: {target_row['Gyro_X']:.2f} | Y: {target_row['Gyro_Y']:.2f} | Z: {target_row['Gyro_Z']:.2f}")

        with m4:
            st.metric(
                label="Decision Priority Check",
                value="Rule 1: Turn" if explanation["is_gyro_high"] else ("Rule 2: Sudden" if explanation["is_acc_change_high"] else "Rule 3: Normal"),
                help="Project rule hierarchy checks Gyroscope > Threshold first, then Acceleration Change > Threshold."
            )
            st.caption(f"Rotational Threshold: **{gyro_threshold:.3f} rad/s**")

        st.markdown(
            f"""
            <div class="explanation-box">
                <div class="explanation-title">📝 Step-by-Step Algorithmic Logic</div>
                <div style="color: #cbd5e1; line-height: 1.7; font-size: 0.95rem;">
                    {"<br>".join(explanation['explanation_lines'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# TAB 4: WHAT DO THESE RESULTS MEAN?
# ---------------------------------------------------------
with tab_insights:
    st.markdown("### 💡 What Do These Results Mean?")
    st.caption("Plain-English translation of the mathematical calculations and behavioral classifications.")

    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        st.markdown(
            """
            #### 1. What does a high Acceleration Change indicate?
            - **Physical Interpretation**: A sudden change in acceleration magnitude ($|\\Delta a|$) represents rapid alteration in the vehicle's speed or road surface impulse.
            - **Driving Context**: This typically occurs during **hard braking** (deceleration) or **rapid gas pedal depression** (acceleration), or when traversing severe road bumps and potholes.
            - **Detection Rule**: In this project, whenever $\\Delta a > \\mu + 2\\sigma$ (${acc_threshold:.3f}$ m/s²), it qualifies as an atypical longitudinal event.
            """
        )

        st.markdown(
            """
            #### 2. What does a high Gyroscope Magnitude indicate?
            - **Physical Interpretation**: The gyroscope measures angular velocity around all three axes (roll, pitch, yaw) in radians per second.
            - **Driving Context**: High rotational velocity reflects the vehicle entering a **sharp corner, round-about, U-turn, or aggressive lane change**.
            - **Detection Rule**: In this project, whenever Gyroscope Magnitude exceeds $\\mu + 2\\sigma$ (${gyro_threshold:.3f}$ rad/s), it is categorized as **Turning**.
            """
        )

    with col_ins2:
        st.markdown(
            f"""
            #### 3. Why are thresholds calculated as $\\mu + 2\\sigma$?
            - **Statistical Rationale**: In probability theory, approximately **95.4%** of values in a normal distribution fall within two standard deviations of the mean.
            - **Adaptive Baseline**: By setting the cutoff at $\\text{{Mean}} + 2 \\times \\text{{Std}}$, the model establishes a trip-specific baseline that automatically filters out ordinary cruising fluctuations and isolates genuine outlier maneuvers (the top ~2-5% most aggressive points).
            - **Current Dataset Thresholds**:
              * Acceleration Change Cutoff: **{acc_threshold:.4f} m/s²**
              * Gyroscope Cutoff: **{gyro_threshold:.4f} rad/s**
            """
        )

        st.markdown(
            f"""
            #### 4. Which event dominates and why?
            - **Observation**: **{stats['most_common_class']}** accounts for **{stats['event_percentages'][stats['most_common_class']]:.1f}%** ({stats['event_counts'][stats['most_common_class']]:,} rows) of the dataset.
            - **Real-World Reality**: During everyday driving, vehicles spend the vast majority of time moving steadily or stationary at signals, with sharp turns and heavy braking occurring as discrete, brief events.
            """
        )

    st.markdown("---")
    st.markdown("#### ⚠️ Critical Engineering Caveats & Limitations")
    st.warning(
        """
        - **Phone Rotation vs. Vehicle Turning**: A high gyroscope spike indicates the *phone* rotated. If the driver handles the phone, or if the phone slips inside a mount, the model will classify it as 'Turning' even if the car was driving straight.
        - **Braking vs. Acceleration Ambiguity**: Because the algorithm computes the scalar magnitude difference `|diff(Magnitude)|`, it detects *speed change intensity* without distinguishing whether the vehicle accelerated forward or braked backward.
        - **Indicator vs. Proof**: Sensor readings are statistical indicators for driver behavior profiling and should not be treated as legal or accident proof without video and GPS correlation.
        """
    )

# ---------------------------------------------------------
# TAB 5: ASK QUESTIONS ABOUT MY DATA
# ---------------------------------------------------------
with tab_qa:
    st.markdown("### 💬 Ask Questions About My Data")
    st.caption("Ask natural questions about dataset metrics, thresholds, classifications, and sensor observations. All answers are computed dynamically via deterministic Python logic.")

    sample_queries = [
        "What does my dataset contain?",
        "What is the most common detected event?",
        "How many observations were classified as Normal Movement?",
        "What percentage of observations were classified as Turning?",
        "What are the calculated thresholds?",
        "At which timestamp is the gyroscope magnitude highest?",
        "Why was this observation classified as Sudden Acceleration/Braking?",
        "Summarize the main findings from my uploaded data.",
        "What are the limitations of these results?"
    ]

    st.markdown("**Click a suggested question to run immediately:**")
    clicked_chip = None
    chip_cols = st.columns(3)
    for idx, q_text in enumerate(sample_queries):
        col_target = chip_cols[idx % 3]
        if col_target.button(q_text, key=f"chip_{idx}", use_container_width=True):
            clicked_chip = q_text

    st.markdown("<br>", unsafe_allow_html=True)
    user_query = st.text_input(
        "Or type your own question:",
        value=clicked_chip if clicked_chip else "",
        placeholder="e.g. What are the calculated thresholds?"
    )

    if user_query:
        ans_obj = answer_user_query(
            user_query,
            pipeline_result,
            selected_row_idx=st.session_state.get("last_selected_row", 18)
        )
        
        st.markdown(
            f"""
            <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 1.4rem; margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span style="color: #818cf8; font-weight: 700; font-size: 1rem;">
                        💡 Answer ({ans_obj['category']})
                    </span>
                    <span style="font-size: 0.8rem; color: #64748b;">Deterministic Python Query Engine</span>
                </div>
                <div style="color: #f1f5f9; line-height: 1.65; font-size: 0.95rem;">
                    {ans_obj['answer']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------------------------------------
# TAB 6: EXPORT & TECHNICAL REPORT
# ---------------------------------------------------------
with tab_export:
    st.markdown("### 📥 Export & Technical Reports")
    st.caption("Download the processed sensor datasets with classified events or generate a formatted technical mini-project report.")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # 1. Download Processed CSV
    with exp_col1:
        st.markdown("#### 📄 Processed Sensor Dataset")
        st.write("Contains synchronized readings, calculated magnitudes, changes, and classification labels.")
        csv_buffer = io.StringIO()
        full_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Processed CSV",
            data=csv_buffer.getvalue(),
            file_name="processed_driving_behavior_data.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 2. Download Summary CSV
    with exp_col2:
        st.markdown("#### 📊 Event Classification Summary")
        st.write("Contains total counts, proportions, and threshold parameters.")
        summary_records = []
        for ev, count in stats["event_counts"].items():
            summary_records.append({
                "Event": ev,
                "Count": count,
                "Percentage": f"{stats['event_percentages'][ev]:.2f}%",
                "Threshold Applied": f"{gyro_threshold:.4f} rad/s" if ev == "Turning" else (f"{acc_threshold:.4f} m/s²" if "Acceleration" in ev else "Baseline")
            })
        summary_df = pd.DataFrame(summary_records)
        sum_csv_buffer = io.StringIO()
        summary_df.to_csv(sum_csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Summary CSV",
            data=sum_csv_buffer.getvalue(),
            file_name="event_classification_summary.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 3. Download Full Technical Markdown Report
    with exp_col3:
        st.markdown("#### 📝 Academic Project Report")
        st.write("Academic-grade documentation with formulas, results tables, and limitations.")
        report_text = generate_markdown_report(pipeline_result)
        st.download_button(
            label="⬇️ Download Technical Report (.md)",
            data=report_text,
            file_name="Driving_Behavior_Detection_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 📑 Report Preview")
    with st.expander("Click to Preview Generated Report", expanded=False):
        st.markdown(report_text)
