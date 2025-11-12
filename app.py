# 🚀 Daewoo FastEx - Enhanced Streamlit Dashboard
# Complete working version with perfect prediction integration

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from io import BytesIO
import os
import numpy as np
from datetime import datetime

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
    }
    
    .main-header {
        background: linear-gradient(135deg, #0047ba 0%, #003399 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .kpi-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--daewoo-blue);
        margin-bottom: 1rem;
    }
    
    .kpi-number {
        font-size: 2rem;
        font-weight: bold;
        color: var(--daewoo-blue);
        margin-bottom: 0.5rem;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f0ff 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fff4 0%, #e6ffe6 100%);
        border-left: 4px solid var(--daewoo-green);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fffaf0 0%, #fff5e6 100%);
        border-left: 4px solid var(--daewoo-orange);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #fff0f0 0%, #ffe6e6 100%);
        border-left: 4px solid var(--daewoo-red);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# ✅ HEADER SECTION
# -------------------------------
def render_header():
    col1, col2 = st.columns([1, 4])
    
    with col1:
        # Try to load logo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(current_dir, "assets", "daewoo_logo.png")
        
        if os.path.exists(logo_path):
            st.image(logo_path, width=120)
        else:
            st.markdown("""
            <div style='width:120px; height:120px; background:#0047ba; color:white; 
                        display:flex; align-items:center; justify-content:center; 
                        border-radius:10px; font-weight:bold; font-size:14px; text-align:center;'>
                DAE WOO<br>FASTEX
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='main-header'>
            <h1 style='margin:0; font-size:2.5em;'>🚚 Daewoo FastEx Analytics</h1>
            <p style='margin:0; font-size:1.2em; opacity:0.9;'>Intelligent Shipment Performance & AI Prediction Platform</p>
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
        
        # Check if all required files exist
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
            
            st.sidebar.error(f"❌ Missing model files: {', '.join(missing_files)}")
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
        text_columns = ['OriginCity', 'DestinationCity', 'ServiceType', 'DeliveryMode', 'Status']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.title()
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def calculate_kpis(df):
    """Calculate key performance indicators"""
    kpis = {}
    
    kpis['total_shipments'] = len(df)
    kpis['total_revenue'] = df['Price'].sum() if 'Price' in df.columns else 0
    kpis['avg_weight'] = df['WeightKg'].mean() if 'WeightKg' in df.columns else 0
    kpis['avg_price'] = df['Price'].mean() if 'Price' in df.columns else 0
    
    # Calculate delivery performance
    if all(col in df.columns for col in ['DeliveryDate', 'PickupDate']):
        df['delivery_days'] = (df['DeliveryDate'] - df['PickupDate']).dt.days
        kpis['avg_delivery_days'] = df['delivery_days'].mean()
    
    # Calculate status counts
    if 'Status' in df.columns:
        kpis['on_time'] = len(df[df['Status'].str.contains('Delivered|Completed', case=False, na=False)])
        kpis['delayed'] = len(df[df['Status'].str.contains('Delay|Late', case=False, na=False)])
    else:
        kpis['on_time'] = 0
        kpis['delayed'] = 0
    
    kpis['on_time_rate'] = (kpis['on_time'] / kpis['total_shipments'] * 100) if kpis['total_shipments'] > 0 else 0
    
    return kpis

# -------------------------------
# ✅ PREDICTION INTERFACE
# -------------------------------
def show_prediction_interface(model, feature_names, model_artifacts):
    """Show the prediction interface with correct feature mapping"""
    st.markdown("---")
    st.header("🔮 AI Delivery Delay Prediction")
    st.markdown("Predict potential delivery delays using our trained machine learning model")
    
    with st.container():
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        with st.form("prediction_form"):
            st.subheader("📋 Enter Shipment Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                weight = st.number_input("📦 Weight (kg)", min_value=0.1, max_value=1000.0, value=25.0, step=0.1, help="Package weight in kilograms")
                pickup_hour = st.slider("⏰ Pickup Hour", 0, 23, 12, help="Hour of the day for pickup")
            
            with col2:
                price = st.number_input("💰 Price (PKR)", min_value=10.0, max_value=50000.0, value=1500.0, step=50.0, help="Shipping cost in Pakistani Rupees")
                booking_month = st.slider("📅 Booking Month", 1, 12, 6, help="Month when booking was made")
            
            with col3:
                service_options = model_artifacts.get('service_categories', ['Economy', 'Express', 'Overnight', 'Same Day'])
                delivery_options = model_artifacts.get('delivery_categories', ['Bus', 'Ground', 'Air'])
                
                service_type = st.selectbox("🚚 Service Type", options=service_options, index=0)
                delivery_mode = st.selectbox("📦 Delivery Mode", options=delivery_options, index=0)
                is_inter_city = st.checkbox("🏙️ Inter-City Shipment", value=True, help="Check if shipment is between different cities")
            
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
                        input_data['FromMajorCity'] = [1]  # Default assumption
                        input_data['ToMajorCity'] = [1]    # Default assumption
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
                        features_df = features_df[feature_names]  # Critical: maintain feature order
                        
                        # Make prediction
                        prediction = model.predict(features_df)[0]
                        prediction_proba = model.predict_proba(features_df)[0]
                        
                        confidence = max(prediction_proba) * 100
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        if prediction == 1:
                            st.markdown('<div class="danger-box">', unsafe_allow_html=True)
                            st.error("## 🚨 HIGH RISK OF DELAY")
                            st.write(f"**Confidence Level:** {confidence:.1f}%")
                            st.progress(int(confidence) / 100)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("Delay Probability", f"{prediction_proba[1]:.1%}")
                            with col_b:
                                st.metric("On-Time Probability", f"{prediction_proba[0]:.1%}")
                            
                            st.warning("""
                            **🚚 Recommended Actions:**
                            - Consider expedited shipping option
                            - Schedule for early morning pickup (6-8 AM)
                            - Assign to experienced driver team
                            - Enable real-time tracking and monitoring
                            - Prepare customer communication plan
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="success-box">', unsafe_allow_html=True)
                            st.success("## ✅ ON-TIME DELIVERY EXPECTED")
                            st.write(f"**Confidence Level:** {confidence:.1f}%")
                            st.progress(int(confidence) / 100)
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.metric("On-Time Probability", f"{prediction_proba[0]:.1%}")
                            with col_b:
                                st.metric("Delay Probability", f"{prediction_proba[1]:.1%}")
                            
                            st.info("""
                            **📦 Status Overview:**
                            - Shipment is likely to arrive as scheduled
                            - Standard monitoring procedures apply
                            - Customer can expect timely delivery
                            - Continue with regular operational workflow
                            """)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Feature importance insight
                        with st.expander("🔍 AI Insights & Factors"):
                            st.write("**Key factors influencing this prediction:**")
                            factors = [
                                ("Service Type", "High impact on delivery timeline"),
                                ("Pickup Time", "Affects routing and scheduling"),
                                ("Weight", "Impacts handling and transit time"),
                                ("Delivery Mode", "Determines transportation method"),
                                ("Inter-City", "Affects distance and route complexity")
                            ]
                            
                            for factor, impact in factors:
                                st.write(f"• **{factor}**: {impact}")
                    
                    except Exception as e:
                        st.error(f"❌ Prediction error: {str(e)}")
                        st.info("Please ensure all model files are properly generated by running the notebook.")
        
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
        st.sidebar.info("""
        **To enable AI predictions:**
        1. Run `fastex_analysis.ipynb`
        2. Execute all cells
        3. Restart this app
        """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Tools")
    
    if st.sidebar.button("🔄 Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared!")
    
    st.sidebar.markdown("---")
    st.sidebar.header("📞 Support")
    st.sidebar.markdown("""
    **Need help?**
    - 📧 analytics@daewoofastex.com
    - 📱 +92-XXX-XXXXXXX
    - 🏢 Daewoo Express HQ
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
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                origin_cities = sorted(df['OriginCity'].unique()) if 'OriginCity' in df.columns else []
                selected_origin = st.multiselect("🏙️ Origin City", origin_cities, default=origin_cities[:3] if origin_cities else [])
            
            with col2:
                dest_cities = sorted(df['DestinationCity'].unique()) if 'DestinationCity' in df.columns else []
                selected_dest = st.multiselect("🎯 Destination City", dest_cities, default=dest_cities[:3] if dest_cities else [])
            
            with col3:
                service_types = sorted(df['ServiceType'].unique()) if 'ServiceType' in df.columns else []
                selected_service = st.multiselect("🚚 Service Type", service_types, default=service_types)
            
            # Apply filters
            filtered_df = df.copy()
            if selected_origin:
                filtered_df = filtered_df[filtered_df['OriginCity'].isin(selected_origin)]
            if selected_dest:
                filtered_df = filtered_df[filtered_df['DestinationCity'].isin(selected_dest)]
            if selected_service:
                filtered_df = filtered_df[filtered_df['ServiceType'].isin(selected_service)]
            
            filtered_kpis = calculate_kpis(filtered_df)
            
            # KPI Dashboard
            st.header("📊 Performance Dashboard")
            
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            
            with kpi1:
                st.metric("Total Shipments", f"{filtered_kpis['total_shipments']:,}")
            with kpi2:
                st.metric("Total Revenue", f"₹{filtered_kpis['total_revenue']:,.0f}")
            with kpi3:
                st.metric("Avg Weight", f"{filtered_kpis['avg_weight']:.1f} kg")
            with kpi4:
                st.metric("On-Time Rate", f"{filtered_kpis['on_time_rate']:.1f}%")
            with kpi5:
                delivery_days = filtered_kpis.get('avg_delivery_days', 'N/A')
                st.metric("Avg Delivery Days", f"{delivery_days:.1f}" if delivery_days != 'N/A' else "N/A")
            
            # Visualizations
            st.header("📈 Analytics & Insights")
            
            viz1, viz2 = st.columns(2)
            
            with viz1:
                if 'ServiceType' in filtered_df.columns:
                    service_counts = filtered_df['ServiceType'].value_counts()
                    fig1 = px.pie(values=service_counts.values, names=service_counts.index, 
                                title="📦 Shipments by Service Type", hole=0.4)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with viz2:
                if 'OriginCity' in filtered_df.columns:
                    city_counts = filtered_df['OriginCity'].value_counts().head(10)
                    fig2 = px.bar(x=city_counts.values, y=city_counts.index, orientation='h',
                                title="🏙️ Top Origin Cities", labels={'x': 'Shipments', 'y': 'City'})
                    st.plotly_chart(fig2, use_container_width=True)
            
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
                        'Metric': ['Total Shipments', 'Total Revenue', 'Average Weight', 'On-Time Rate', 'Average Delivery Days'],
                        'Value': [
                            filtered_kpis['total_shipments'],
                            f"₹{filtered_kpis['total_revenue']:,.2f}",
                            f"{filtered_kpis['avg_weight']:.2f} kg",
                            f"{filtered_kpis['on_time_rate']:.2f}%",
                            f"{filtered_kpis.get('avg_delivery_days', 'N/A')}"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
                
                return output.getvalue()
            
            excel_data = convert_to_excel(filtered_df)
            
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_data,
                file_name=f"daewoo_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Data Preview
            with st.expander("🔍 Preview Filtered Data"):
                st.dataframe(filtered_df.head(100), use_container_width=True)
                st.caption(f"Showing {min(100, len(filtered_df))} of {len(filtered_df):,} records")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to Daewoo FastEx Analytics Platform
        
        ### 🚀 Get Started:
        1. **Run the Notebook** → Execute `fastex_analysis.ipynb` to train AI models
        2. **Upload Your Data** → Use the sidebar to upload shipment data
        3. **Analyze Performance** → View interactive dashboards and KPIs
        4. **AI Predictions** → Get intelligent delay forecasts
        5. **Export Reports** → Download customized analytics reports
        
        ### 📊 Supported Data Format:
        Your Excel/CSV should include:
        - `TrackingID`, `BookingDate`, `PickupDate`, `DeliveryDate`
        - `OriginCity`, `DestinationCity`, `ServiceType`, `DeliveryMode`  
        - `WeightKg`, `Price`, `Status`, `DelayReason` (optional)
        
        ### 🎯 Key Features:
        - **Real-time Analytics** - Interactive dashboards
        - **AI Delay Prediction** - Machine learning forecasts
        - **Performance KPIs** - Key metrics and trends
        - **Data Export** - Excel reports with summaries
        - **Smart Filtering** - Segment data by multiple criteria
        """)
        
        # Quick stats
        if prediction_model:
            st.success("✅ AI Prediction model is loaded and ready!")
        else:
            st.warning("⚠️ Run the notebook first to enable AI delay predictions")

# -------------------------------
# ✅ RUN APPLICATION
# -------------------------------
if __name__ == "__main__":
    main()