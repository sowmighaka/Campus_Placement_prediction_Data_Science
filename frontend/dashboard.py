import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(page_title="AI Campus Placement Predictor", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #4F8EF7 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Data and Models
@st.cache_resource
def load_artifacts():
    base_path = os.path.dirname(__file__)
    backend_path = os.path.join(base_path, "..", "backend")
    
    models = joblib.load(os.path.join(backend_path, "models", "models.pkl"))
    scaler = joblib.load(os.path.join(backend_path, "models", "scaler.pkl"))
    ohe = joblib.load(os.path.join(backend_path, "models", "encoder.pkl"))
    features = joblib.load(os.path.join(backend_path, "models", "features.pkl"))
    df = pd.read_csv(os.path.join(backend_path, "data", "student_placement_data.csv"))
    return models, scaler, ohe, features, df

models, scaler, ohe, features, df = load_artifacts()

# Sidebar for Input
st.sidebar.header("🎓 Student Information")

with st.sidebar:
    ssc_p = st.slider("10th Percentage", 50.0, 100.0, 75.0)
    hsc_p = st.slider("12th Percentage", 50.0, 100.0, 75.0)
    cgpa = st.slider("UG CGPA", 5.0, 10.0, 7.5)
    backlogs = st.number_input("Number of Backlogs", 0, 10, 0)
    branch = st.selectbox("Branch", ["CE", "CSE", "ECE", "EE", "ME"])
    prog_skill = st.select_slider("Programming Skill Level", options=range(1, 11), value=5)
    projects = st.number_input("Number of Projects", 0, 10, 2)
    internship = st.radio("Internship Experience", ["No", "Yes"])
    aptitude = st.slider("Aptitude Score", 0.0, 100.0, 60.0)
    comm_skill = st.select_slider("Communication Skills (1-10)", options=range(1, 11), value=6)
    mock_interviews = st.number_input("Mock Interviews Attended", 0, 20, 1)
    
    # selected_model_name = st.selectbox("Select ML Model", list(models.keys()))
    selected_model_name = "Random Forest" # Default to Random Forest as per user request to simplify

# Main Header
st.title("🚀 Campus Placement Prediction Dashboard")
st.markdown("Enter your details in the sidebar to see real-time placement probability and insights.")

# Prepare numeric data
numeric_df = pd.DataFrame({
    'ssc_percentage': [ssc_p],
    'hsc_percentage': [hsc_p],
    'cgpa': [cgpa],
    'backlogs': [backlogs],
    'programming_skill': [prog_skill],
    'projects': [projects],
    'internship': [1 if internship == "Yes" else 0],
    'aptitude_score': [aptitude],
    'communication_skills': [comm_skill],
    'mock_interviews': [mock_interviews]
})

# Handle Branch Encoding (One-Hot)
branch_encoded = ohe.transform([[branch]])
branch_columns = ohe.get_feature_names_out(['branch'])
branch_df = pd.DataFrame(branch_encoded, columns=branch_columns)

# Combine numeric and encoded features
input_df = pd.concat([numeric_df, branch_df], axis=1)

# Ensure feature order matches the training data
input_df = input_df[features]

# Scaling
input_scaled = scaler.transform(input_df)

# Prediction
selected_model = models[selected_model_name]
prob = selected_model.predict_proba(input_scaled)[0][1]
prediction = selected_model.predict(input_scaled)[0]

# --- Layout ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Placement Probability")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Placement Probability", 'font': {'size': 24, 'color': '#4F8EF7', 'family': 'Outfit'}},
        number = {'font': {'size': 80, 'color': 'white', 'family': 'Outfit'}, 'suffix': '%'},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "white", 'thickness': 0.2},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#ff3e3e'},
                {'range': [40, 70], 'color': '#ffaa00'},
                {'range': [70, 100], 'color': '#00f2fe'}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100}}))
    
    fig_gauge.update_layout(
        margin=dict(t=50, b=0, l=30, r=30),
        paper_bgcolor = 'rgba(0,0,0,0)', 
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = {'color': "white", 'family': "Outfit"}
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    if prediction == 1:
        st.success(f"🎉 Result: Highly Likely to be Placed! ({selected_model_name})")
    else:
        st.warning(f"⚠️ Result: Low Probability of Placement. ({selected_model_name})")

with col2:
    st.subheader("User vs Dataset Comparison")
    
    # Placement Sensitivity (Line Chart)
    # How probability changes as CGPA varies (keeping other inputs same)
    cgpa_range = np.linspace(5.0, 10.0, 50)
    
    # Create a batch of inputs for sensitivity analysis
    sensitivity_df = pd.concat([input_df] * len(cgpa_range), ignore_index=True)
    sensitivity_df['cgpa'] = cgpa_range
    
    # Scale and predict
    sensitivity_scaled = scaler.transform(sensitivity_df)
    probs = selected_model.predict_proba(sensitivity_scaled)[:, 1]
    
    line_df = pd.DataFrame({
        'Potential CGPA': cgpa_range,
        'Placement Probability': probs * 100
    })
    
    fig_line = px.line(line_df, x='Potential CGPA', y='Placement Probability',
                       title="Sensitivity Analysis: Impact of CGPA on Placement Chance",
                       color_discrete_sequence=['#00f2fe'])
    
    # Add marker for current CGPA
    fig_line.add_trace(go.Scatter(
        x=[cgpa], y=[prob * 100],
        mode='markers',
        marker=dict(size=12, color='#4F8EF7', symbol='circle', line=dict(width=2, color='white')),
        name='Current Status'
    ))

    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        yaxis_range=[0, 100],
        showlegend=True
    )
    
    st.plotly_chart(fig_line, use_container_width=True)
    st.info("💡 This line shows how your placement probability would change if your CGPA was different, holding all other factors constant.")

# --- Second Row ---
st.divider()

st.subheader("Feature Importance")
if hasattr(selected_model, 'feature_importances_'):
    importances = selected_model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig_imp = px.bar(feat_imp_df, y='Feature', x='Importance', orientation='h',
                     title=f"What influenced this prediction? ({selected_model_name})")
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    # For Logistic Regression, use coefficients
    coeffs = np.abs(selected_model.coef_[0])
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': coeffs}).sort_values('Importance', ascending=True)
    fig_imp = px.bar(feat_imp_df, y='Feature', x='Importance', orientation='h',
                     title=f"Feature Impact (Logistic Regression Coefficients)")
    st.plotly_chart(fig_imp, use_container_width=True)

# --- Insights ---
st.subheader("💡 Personal Insights")
insights = []
avg_cgpa = df['cgpa'].mean()
if cgpa < avg_cgpa:
    insights.append(f"• Your CGPA ({cgpa:.2f}) is slightly below average ({avg_cgpa:.2f}). Consider focusing on technical projects to compensate.")
else:
    insights.append(f"• Your CGPA ({cgpa:.2f}) is above average. This is a strong positive factor for placement.")

if prog_skill > 7:
    insights.append("• Your strong programming skills are a significant advantage in the current market.")
elif prog_skill < 5:
    insights.append("• Consider improving your programming skills; many companies prioritize coding proficiency.")

if backlogs > 0:
    insights.append(f"• Having {backlogs} backlogs may reduce your chances at Tier-1 companies. Focus on clearing them.")

if internship == "No":
    insights.append("• Lack of internship experience might be a hurdle. Try to gain some practical exposure or work on live projects.")

for insight in insights:
    st.markdown(insight)

st.sidebar.markdown("---")
st.sidebar.info("This dashboard uses real-time ML predictions to provide placement insights.")
