# 🚀 Daewoo FastEx - Enhanced Analytics Dashboard
# Complete with comprehensive visualizations

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import joblib
from io import BytesIO
import os
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# ✅ PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Daewoo FastEx Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# ✅ ENHANCED CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --daewoo-blue: #0047ba;
        --daewoo-light-blue: #e6f0ff;
        --daewoo-dark-blue: #003399;
        --daewoo-green: #00a650;
        --daewoo-orange: #ff6b00;
        --daewoo-red: #ff4444;
        --daewoo-purple: #8a2be2;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0047ba 0%, #003399 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 71, 186, 0.3);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--daewoo-blue);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f0ff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid var(--daewoo-light-blue);
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 71, 186, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ HEADER SECTION
# -------------------------------
def render_header():
    col1, col2 = st.columns([1, 4])
    
    with col1:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, "assets", "daewoo_logo.png")
        
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)
        else:
            st.markdown("""
            <div style='width:120px; height:120px; background:#0047ba; color:white; 
                        display:flex; align-items:center; justify-content:center; 
                        border-radius:12px; font-weight:bold; font-size:14px; text-align:center;'>
                DAE WOO<br>FASTEX
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0; font-size:2.8em;'>🚚 Daewoo FastEx Analytics</h1>
            <p style='margin:0; font-size:1.3em; opacity:0.9;'>Comprehensive Logistics Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)

render_header()

# -------------------------------
# ✅ MODEL LOADING
# -------------------------------
@st.cache_resource
def load_prediction_model():
    """Load the trained prediction model and artifacts"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        model_path = os.path.join(current_dir, "outputs", "rf_delay_model.joblib")
        artifacts_path = os.path.join(current_dir, "outputs", "model_artifacts.joblib")
        feature_names_path = os.path.join(current_dir, "outputs", "feature_names.joblib")
        
        if all(os.path.exists(path) for path in [model_path, artifacts_path, feature_names_path]):
            model = joblib.load(model_path)
            artifacts = joblib.load(artifacts_path)
            feature_names = joblib.load(feature_names_path)
            
            st.sidebar.success("✅ AI Model loaded successfully!")
            return model, feature_names, artifacts
        else:
            missing_files = []
            if not os.path.exists(model_path): missing_files.append("rf_delay_model.joblib")
            if not os.path.exists(artifacts_path): missing_files.append("model_artifacts.joblib")
            if not os.path.exists(feature_names_path): missing_files.append("feature_names.joblib")
            
            st.sidebar.error(f"❌ Missing: {', '.join(missing_files)}")
            return None, [], {}
            
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None, [], {}

# -------------------------------
# ✅ DATA PROCESSING FUNCTIONS
# -------------------------------
@st.cache_data
def load_data(uploaded_file):
    """Load and process uploaded data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload CSV or Excel.")
            return None
        
        st.success(f"✅ Data loaded successfully: {len(df):,} records")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert date columns
        date_columns = ['BookingDate', 'PickupDate', 'DeliveryDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['OriginCity', 'DestinationCity', 'ServiceType', 'DeliveryMode', 'Status', 'PackageType', 'DelayReason']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        # Extract numeric values from mixed columns
        if 'WeightKg' in df.columns:
            df['WeightKg'] = pd.to_numeric(df['WeightKg'].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
        
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.extract('(\d+\.?\d*)')[0], errors='coerce')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate comprehensive key performance indicators"""
    kpis = {}
    
    kpis['total_shipments'] = len(df)
    kpis['total_revenue'] = df['Price'].sum() if 'Price' in df.columns else 0
    kpis['avg_weight'] = df['WeightKg'].mean() if 'WeightKg' in df.columns else 0
    kpis['avg_price'] = df['Price'].mean() if 'Price' in df.columns else 0
    kpis['max_weight'] = df['WeightKg'].max() if 'WeightKg' in df.columns else 0
    kpis['min_weight'] = df['WeightKg'].min() if 'WeightKg' in df.columns else 0
    
    # Calculate delivery performance
    if all(col in df.columns for col in ['DeliveryDate', 'PickupDate']):
        df['delivery_days'] = (df['DeliveryDate'] - df['PickupDate']).dt.days
        kpis['avg_delivery_days'] = df['delivery_days'].mean()
        kpis['max_delivery_days'] = df['delivery_days'].max()
        kpis['min_delivery_days'] = df['delivery_days'].min()
    
    # Calculate status counts
    if 'Status' in df.columns:
        kpis['on_time'] = len(df[df['Status'].str.contains('Delivered|Completed', case=False, na=False)])
        kpis['delayed'] = len(df[df['Status'].str.contains('Delay|Late', case=False, na=False)])
        kpis['in_transit'] = len(df[df['Status'].str.contains('Transit|Processing', case=False, na=False)])
    else:
        kpis['on_time'] = 0
        kpis['delayed'] = 0
        kpis['in_transit'] = 0
    
    kpis['on_time_rate'] = (kpis['on_time'] / kpis['total_shipments'] * 100) if kpis['total_shipments'] > 0 else 0
    kpis['delay_rate'] = (kpis['delayed'] / kpis['total_shipments'] * 100) if kpis['total_shipments'] > 0 else 0
    
    # City statistics
    if 'OriginCity' in df.columns:
        kpis['unique_origins'] = df['OriginCity'].nunique()
        kpis['top_origin'] = df['OriginCity'].mode()[0] if len(df['OriginCity'].mode()) > 0 else 'N/A'
    
    if 'DestinationCity' in df.columns:
        kpis['unique_destinations'] = df['DestinationCity'].nunique()
        kpis['top_destination'] = df['DestinationCity'].mode()[0] if len(df['DestinationCity'].mode()) > 0 else 'N/A'
    
    return kpis

# -------------------------------
# ✅ COMPREHENSIVE VISUALIZATIONS
# -------------------------------
def create_comprehensive_visualizations(df, filtered_df):
    """Create all possible visualizations for the data"""
    
    # 1. SERVICE TYPE ANALYSIS
    st.markdown("### 📦 Service Type Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'ServiceType' in filtered_df.columns:
            service_counts = filtered_df['ServiceType'].value_counts()
            fig_service_pie = px.pie(
                values=service_counts.values,
                names=service_counts.index,
                title="Service Type Distribution",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_service_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_service_pie, use_container_width=True)
    
    with col2:
        if 'ServiceType' in filtered_df.columns and 'Price' in filtered_df.columns:
            service_price = filtered_df.groupby('ServiceType')['Price'].mean().sort_values(ascending=False)
            fig_service_price = px.bar(
                x=service_price.index,
                y=service_price.values,
                title="Average Price by Service Type",
                labels={'x': 'Service Type', 'y': 'Average Price'},
                color=service_price.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_service_price, use_container_width=True)
    
    # 2. GEOGRAPHICAL ANALYSIS
    st.markdown("### 🗺️ Geographical Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'OriginCity' in filtered_df.columns:
            origin_counts = filtered_df['OriginCity'].value_counts().head(10)
            fig_origin = px.bar(
                x=origin_counts.values,
                y=origin_counts.index,
                orientation='h',
                title="Top 10 Origin Cities",
                labels={'x': 'Number of Shipments', 'y': 'City'},
                color=origin_counts.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_origin, use_container_width=True)
    
    with col2:
        if 'DestinationCity' in filtered_df.columns:
            dest_counts = filtered_df['DestinationCity'].value_counts().head(10)
            fig_dest = px.bar(
                x=dest_counts.values,
                y=dest_counts.index,
                orientation='h',
                title="Top 10 Destination Cities",
                labels={'x': 'Number of Shipments', 'y': 'City'},
                color=dest_counts.values,
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_dest, use_container_width=True)
    
    # 3. ROUTE ANALYSIS
    st.markdown("### 🚛 Route Analysis")
    if all(col in filtered_df.columns for col in ['OriginCity', 'DestinationCity']):
        route_counts = filtered_df.groupby(['OriginCity', 'DestinationCity']).size().reset_index(name='Count')
        top_routes = route_counts.nlargest(10, 'Count')
        
        fig_routes = px.bar(
            top_routes,
            x='Count',
            y=top_routes.apply(lambda x: f"{x['OriginCity']} → {x['DestinationCity']}", axis=1),
            orientation='h',
            title="Top 10 Busiest Routes",
            labels={'x': 'Number of Shipments', 'y': 'Route'},
            color='Count',
            color_continuous_scale='Rainbow'
        )
        st.plotly_chart(fig_routes, use_container_width=True)
    
    # 4. PRICE & WEIGHT ANALYSIS
    st.markdown("### 💰 Price & Weight Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Price' in filtered_df.columns:
            fig_price_dist = px.histogram(
                filtered_df,
                x='Price',
                nbins=30,
                title="Price Distribution",
                color_discrete_sequence=['#0047ba']
            )
            fig_price_dist.update_layout(bargap=0.1)
            st.plotly_chart(fig_price_dist, use_container_width=True)
    
    with col2:
        if 'WeightKg' in filtered_df.columns:
            fig_weight_dist = px.histogram(
                filtered_df,
                x='WeightKg',
                nbins=30,
                title="Weight Distribution (Kg)",
                color_discrete_sequence=['#00a650']
            )
            fig_weight_dist.update_layout(bargap=0.1)
            st.plotly_chart(fig_weight_dist, use_container_width=True)
    
    # 5. DELIVERY PERFORMANCE
    st.markdown("### ⏱️ Delivery Performance")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Status' in filtered_df.columns:
            status_counts = filtered_df['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Delivery Status Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_status, use_container_width=True)
    
    with col2:
        if 'Status' in filtered_df.columns and 'ServiceType' in filtered_df.columns:
            delay_by_service = filtered_df[filtered_df['Status'].str.contains('Delay', case=False, na=False)]
            if not delay_by_service.empty:
                service_delays = delay_by_service['ServiceType'].value_counts()
                fig_service_delays = px.bar(
                    x=service_delays.index,
                    y=service_delays.values,
                    title="Delays by Service Type",
                    labels={'x': 'Service Type', 'y': 'Number of Delays'},
                    color=service_delays.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_service_delays, use_container_width=True)
    
    # 6. TEMPORAL ANALYSIS
    st.markdown("### 📅 Temporal Analysis")
    
    if 'BookingDate' in filtered_df.columns:
        # Daily trends
        filtered_df['BookingDate'] = pd.to_datetime(filtered_df['BookingDate'])
        daily_trends = filtered_df.groupby(filtered_df['BookingDate'].dt.date).agg({
            'Price': 'sum',
            'TrackingID': 'count'
        }).reset_index()
        daily_trends.columns = ['Date', 'Daily Revenue', 'Daily Shipments']
        
        fig_temporal = sp.make_subplots(specs=[[{"secondary_y": True}]])
        
        # Revenue line
        fig_temporal.add_trace(
            go.Scatter(x=daily_trends['Date'], y=daily_trends['Daily Revenue'], 
                      name="Daily Revenue", line=dict(color='#0047ba', width=3)),
            secondary_y=False,
        )
        
        # Shipments line
        fig_temporal.add_trace(
            go.Scatter(x=daily_trends['Date'], y=daily_trends['Daily Shipments'],
                      name="Daily Shipments", line=dict(color='#00a650', width=3)),
            secondary_y=True,
        )
        
        fig_temporal.update_layout(
            title="Daily Revenue & Shipment Trends",
            xaxis_title="Date",
            hovermode='x unified'
        )
        
        fig_temporal.update_yaxes(title_text="Revenue", secondary_y=False)
        fig_temporal.update_yaxes(title_text="Shipments", secondary_y=True)
        
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # 7. DELIVERY MODE ANALYSIS
    st.markdown("### 🚚 Delivery Mode Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'DeliveryMode' in filtered_df.columns:
            mode_counts = filtered_df['DeliveryMode'].value_counts()
            fig_mode = px.pie(
                values=mode_counts.values,
                names=mode_counts.index,
                title="Delivery Mode Distribution",
                hole=0.3,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_mode, use_container_width=True)
    
    with col2:
        if 'DeliveryMode' in filtered_df.columns and 'Price' in filtered_df.columns:
            mode_price = filtered_df.groupby('DeliveryMode')['Price'].mean()
            fig_mode_price = px.bar(
                x=mode_price.index,
                y=mode_price.values,
                title="Average Price by Delivery Mode",
                labels={'x': 'Delivery Mode', 'y': 'Average Price'},
                color=mode_price.values,
                color_continuous_scale='Purples'
            )
            st.plotly_chart(fig_mode_price, use_container_width=True)
    
    # 8. PACKAGE TYPE ANALYSIS
    if 'PackageType' in filtered_df.columns:
        st.markdown("### 📦 Package Type Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            package_counts = filtered_df['PackageType'].value_counts()
            fig_package = px.pie(
                values=package_counts.values,
                names=package_counts.index,
                title="Package Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_package, use_container_width=True)
        
        with col2:
            if 'WeightKg' in filtered_df.columns:
                package_weight = filtered_df.groupby('PackageType')['WeightKg'].mean()
                fig_package_weight = px.bar(
                    x=package_weight.index,
                    y=package_weight.values,
                    title="Average Weight by Package Type",
                    labels={'x': 'Package Type', 'y': 'Average Weight (Kg)'},
                    color=package_weight.values,
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_package_weight, use_container_width=True)
    
    # 9. DELAY REASON ANALYSIS
    if 'DelayReason' in filtered_df.columns:
        st.markdown("### ⚠️ Delay Reason Analysis")
        delay_reasons = filtered_df[filtered_df['DelayReason'].str.contains('None|nan', case=False, na=True) == False]
        if not delay_reasons.empty and 'DelayReason' in delay_reasons.columns:
            reason_counts = delay_reasons['DelayReason'].value_counts()
            fig_reasons = px.bar(
                x=reason_counts.values,
                y=reason_counts.index,
                orientation='h',
                title="Delay Reasons Analysis",
                labels={'x': 'Number of Delays', 'y': 'Reason'},
                color=reason_counts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_reasons, use_container_width=True)

# -------------------------------
# ✅ PREDICTION INTERFACE
# -------------------------------
def show_prediction_interface(model, feature_names, model_artifacts):
    """Show the prediction interface"""
    st.markdown("---")
    st.header("🔮 AI Delivery Delay Prediction")
    
    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.subheader("📋 Enter Shipment Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weight = st.number_input("📦 Weight (kg)", min_value=0.1, max_value=1000.0, value=25.0, step=0.1)
                pickup_hour = st.slider("⏰ Pickup Hour", 0, 23, 12)
            
            with col2:
                price = st.number_input("💰 Price (PKR)", min_value=10.0, max_value=50000.0, value=1500.0, step=50.0)
                booking_month = st.slider("📅 Booking Month", 1, 12, 6)
            
            with col3:
                service_options = model_artifacts.get('service_categories', ['Economy', 'Express', 'Overnight', 'Same Day'])
                delivery_options = model_artifacts.get('delivery_categories', ['Bus', 'Ground', 'Air'])
                
                service_type = st.selectbox("🚚 Service Type", options=service_options, index=0)
                delivery_mode = st.selectbox("📦 Delivery Mode", options=delivery_options, index=0)
                is_inter_city = st.checkbox("🏙️ Inter-City Shipment", value=True)
            
            predict_button = st.form_submit_button("🎯 Predict Delivery Status", use_container_width=True)
            
            if predict_button:
                with st.spinner("🤖 Analyzing shipment patterns with AI..."):
                    try:
                        # Create input data with all expected features
                        input_data = {feature: [0] for feature in feature_names}
                        
                        # Set basic features
                        input_data['WeightKg'] = [weight]
                        input_data['PickupHour'] = [pickup_hour]
                        input_data['BookingMonth'] = [booking_month]
                        input_data['IsInterCity'] = [int(is_inter_city)]
                        input_data['FromMajorCity'] = [1]
                        input_data['ToMajorCity'] = [1]
                        input_data['IsPeakHour'] = [int(8 <= pickup_hour <= 18)]
                        
                        # Set one-hot encoded features
                        service_col = f'ServiceType_{service_type}'
                        delivery_col = f'DeliveryMode_{delivery_mode}'
                        
                        if service_col in input_data:
                            input_data[service_col] = [1]
                        if delivery_col in input_data:
                            input_data[delivery_col] = [1]
                        
                        # Create feature DataFrame with correct order
                        features_df = pd.DataFrame(input_data)
                        features_df = features_df[feature_names]
                        
                        # Make prediction
                        prediction = model.predict(features_df)[0]
                        prediction_proba = model.predict_proba(features_df)[0]
                        
                        confidence = max(prediction_proba) * 100
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        if prediction == 1:
                            st.error("## 🚨 HIGH RISK OF DELAY")
                            st.write(f"**Confidence Level:** {confidence:.1f}%")
                            st.progress(int(confidence) / 100)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Delay Probability", f"{prediction_proba[1]:.1%}")
                            with col_b:
                                st.metric("On-Time Probability", f"{prediction_proba[0]:.1%}")
                            
                            st.warning("**🚚 Recommended Actions:** Consider expedited shipping, early pickup scheduling, and close monitoring.")
                        else:
                            st.success("## ✅ ON-TIME DELIVERY EXPECTED")
                            st.write(f"**Confidence Level:** {confidence:.1f}%")
                            st.progress(int(confidence) / 100)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("On-Time Probability", f"{prediction_proba[0]:.1%}")
                            with col_b:
                                st.metric("Delay Probability", f"{prediction_proba[1]:.1%}")
                            
                            st.info("**📦 Status Overview:** Shipment is likely to arrive as scheduled with standard monitoring.")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# ✅ SIDEBAR
# -------------------------------
def render_sidebar():
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Shipment Data", 
        type=['csv', 'xlsx'],
        help="Upload your CSV or Excel file with shipment records"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("ℹ️ System Info")
    
    # Model status
    model, feature_names, artifacts = load_prediction_model()
    if model:
        st.sidebar.success("AI Model: ✅ Loaded")
        st.sidebar.write(f"Features: {len(feature_names)}")
        if 'model_accuracy' in artifacts:
            st.sidebar.write(f"Accuracy: {artifacts['model_accuracy']:.1%}")
    else:
        st.sidebar.error("AI Model: ❌ Not Found")
        st.sidebar.info("Run the notebook first to generate AI models")
    
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Tools")
    
    if st.sidebar.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📞 Support")
    st.sidebar.markdown("""
    **Need help?**
    - 📧 Daewoofastexsukkur@gmail.com
    - 📱 +92-345-3874327
    - 📘 Facebook.com/Daewoo FastEx
    """)
    
    return uploaded_file, model, feature_names, artifacts

# -------------------------------
# ✅ MAIN APPLICATION
# -------------------------------
def main():
    # Render sidebar and load model
    uploaded_file, prediction_model, feature_names, model_artifacts = render_sidebar()
    
    if uploaded_file is not None:
        # Load and process data
        df = load_data(uploaded_file)
        
        if df is not None:
            # Calculate KPIs
            kpis = calculate_kpis(df)
            
            # Filters Section
            st.header("🔍 Data Filters & Segmentation")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                origin_cities = sorted(df['OriginCity'].unique()) if 'OriginCity' in df.columns else []
                selected_origin = st.multiselect("🏙️ Origin City", origin_cities, default=origin_cities[:3] if origin_cities else [])
            
            with col2:
                dest_cities = sorted(df['DestinationCity'].unique()) if 'DestinationCity' in df.columns else []
                selected_dest = st.multiselect("🎯 Destination City", dest_cities, default=dest_cities[:3] if dest_cities else [])
            
            with col3:
                service_types = sorted(df['ServiceType'].unique()) if 'ServiceType' in df.columns else []
                selected_service = st.multiselect("🚚 Service Type", service_types, default=service_types)
            
            with col4:
                delivery_modes = sorted(df['DeliveryMode'].unique()) if 'DeliveryMode' in df.columns else []
                selected_delivery = st.multiselect("📦 Delivery Mode", delivery_modes, default=delivery_modes)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_origin:
                filtered_df = filtered_df[filtered_df['OriginCity'].isin(selected_origin)]
            if selected_dest:
                filtered_df = filtered_df[filtered_df['DestinationCity'].isin(selected_dest)]
            if selected_service:
                filtered_df = filtered_df[filtered_df['ServiceType'].isin(selected_service)]
            if selected_delivery:
                filtered_df = filtered_df[filtered_df['DeliveryMode'].isin(selected_delivery)]
            
            filtered_kpis = calculate_kpis(filtered_df)
            
            # KPI Dashboard
            st.header("📊 Performance Dashboard")
            
            # Create metric cards
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem; font-weight:bold; color:#0047ba;">{filtered_kpis['total_shipments']:,}</div>
                    <div style="color:#666; font-weight:500;">Total Shipments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem; font-weight:bold; color:#00a650;">₹{filtered_kpis['total_revenue']:,.0f}</div>
                    <div style="color:#666; font-weight:500;">Total Revenue</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem; font-weight:bold; color:#ff6b00;">{filtered_kpis['on_time_rate']:.1f}%</div>
                    <div style="color:#666; font-weight:500;">On-Time Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem; font-weight:bold; color:#8a2be2;">{filtered_kpis['avg_weight']:.1f} kg</div>
                    <div style="color:#666; font-weight:500;">Avg Weight</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                delivery_days = filtered_kpis.get('avg_delivery_days', 'N/A')
                display_value = f"{delivery_days:.1f}" if delivery_days != 'N/A' else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:2rem; font-weight:bold; color:#0047ba;">{display_value}</div>
                    <div style="color:#666; font-weight:500;">Avg Delivery Days</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Comprehensive Visualizations
            st.markdown("---")
            st.header("📈 Comprehensive Analytics & Insights")
            create_comprehensive_visualizations(df, filtered_df)
            
            # AI Prediction Section
            if prediction_model:
                show_prediction_interface(prediction_model, feature_names, model_artifacts)
            
            # Data Export
            st.header("📥 Export & Reports")
            
            def convert_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Shipment_Data')
                    
                    # Summary sheet
                    summary_data = {
                        'Metric': ['Total Shipments', 'Total Revenue', 'Average Weight', 'On-Time Rate', 'Delay Rate', 'Average Delivery Days'],
                        'Value': [
                            filtered_kpis['total_shipments'],
                            f"₹{filtered_kpis['total_revenue']:,.2f}",
                            f"{filtered_kpis['avg_weight']:.2f} kg",
                            f"{filtered_kpis['on_time_rate']:.2f}%",
                            f"{filtered_kpis['delay_rate']:.2f}%",
                            f"{filtered_kpis.get('avg_delivery_days', 'N/A')}"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
                
                return output.getvalue()
            
            excel_data = convert_to_excel(filtered_df)
            
            st.download_button(
                label="📊 Download Comprehensive Excel Report",
                data=excel_data,
                file_name=f"daewoo_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Data Preview
            with st.expander("🔍 Preview Filtered Data"):
                st.dataframe(filtered_df.head(50), use_container_width=True)
                st.caption(f"Showing {min(50, len(filtered_df))} of {len(filtered_df):,} filtered records")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to Daewoo FastEx Analytics Platform
        
        ### 🚀 Get Started:
        1. **Upload Your Data** → Use the sidebar to upload shipment data (CSV/Excel)
        2. **Explore Analytics** → View comprehensive visualizations and insights
        3. **AI Predictions** → Get intelligent delay forecasts
        4. **Export Reports** → Download customized analytics reports
        
        ### 📊 Supported Data Format:
        Your file should include these columns:
        - `TrackingID`, `BookingDate`, `PickupDate`, `DeliveryDate`
        - `OriginCity`, `DestinationCity`, `ServiceType`, `DeliveryMode`
        - `WeightKg`, `Price`, `Status`, `PackageType`, `DelayReason`
        
        ### 🎯 Comprehensive Analytics Include:
        - Service Type Distribution & Performance
        - Geographical Analysis (Cities & Routes)
        - Price & Weight Distribution
        - Delivery Performance Metrics
        - Temporal Trends Analysis
        - Package Type Insights
        - Delay Reason Analysis
        """)
        
        if prediction_model:
            st.success("✅ AI Prediction model is loaded and ready!")
        else:
            st.warning("⚠️ Run the notebook first to enable AI delay predictions")

# -------------------------------
# ✅ RUN APPLICATION
# -------------------------------
if __name__ == "__main__":
    main()
