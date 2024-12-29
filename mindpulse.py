import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict

# Constants
MINIMUM_AGE = 18
PHQ9_RISK_THRESHOLD = 15

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling/staying asleep, sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself?",
    "Trouble concentrating on things?",
    "Moving or speaking slowly/being fidgety or restless?",
    "Thoughts that you would be better off dead or of hurting yourself?"
]

PHYSIO_METRICS = [
    "Average Heart Rate (bpm)",
    "Heart Rate Variability (ms)",
    "Average Sleep Duration (hours)",
    "Sleep Quality (1-10)",
    "Daily Activity Level (steps)",
    "Stress Level (1-10)"
]

LIFESTYLE_SUGGESTIONS = [
    "Maintain a consistent sleep schedule (7-9 hours daily)",
    "Engage in moderate physical activity (30 minutes, 5 days/week)",
    "Practice mindfulness or meditation (10-15 minutes daily)",
    "Build and maintain social connections through regular interactions",
    "Follow a balanced nutrition plan rich in whole foods"
]

class WellnessAnalytics:
    @staticmethod
    def generate_wellness_radar(physio_data):
        categories = list(physio_data.keys())
        values = list(physio_data.values())
        
        normalized_values = []
        for val in values:
            if isinstance(val, (int, float)):
                normalized_values.append(min(100, max(0, val * 10)))
            else:
                normalized_values.append(50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Current Metrics',
            line=dict(color='#2E8B57', width=3)
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    showline=False,
                    ticks='outside'
                )
            ),
            showlegend=False,
            title={
                'text': 'Wellness Metrics Analysis',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=50)
        )
        
        return fig

def get_assessment(phq9_data: Dict) -> Dict:
    phq9_score = sum(phq9_data.values())
    
    is_depressed = phq9_score >= 10
    severity = "Severe" if phq9_score >= 20 else "Moderately Severe" if phq9_score >= 15 else "Moderate" if phq9_score >= 10 else "Mild" if phq9_score >= 5 else "Minimal"
    
    risk_level = "High" if phq9_score >= PHQ9_RISK_THRESHOLD else "Medium" if phq9_score >= 10 else "Low"
    
    recommendations = []
    if phq9_score >= 15:
        recommendations.extend([
            "Urgent: Schedule appointment with mental health professional",
            "Consider crisis support services if needed",
            "Daily check-ins with trusted support person"
        ])
    elif phq9_score >= 10:
        recommendations.extend([
            "Consult with mental health professional",
            "Implement stress reduction techniques",
            "Regular exercise and sleep schedule"
        ])
    else:
        recommendations.extend([
            "Continue self-monitoring",
            "Maintain healthy lifestyle habits",
            "Practice preventive mental wellness"
        ])
    
    return {
        "is_depressed": is_depressed,
        "severity": severity,
        "risk_level": risk_level,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }

def main():
    st.set_page_config(page_title="MindPulse.ai Analytics", layout="wide")
    
    # Custom CSS
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        .main-title {
            background: linear-gradient(135deg, rgba(46, 139, 87, 0.1), rgba(46, 139, 87, 0.2));
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .result-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #f0f0f0;
        }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
            font-size: 1.1rem;
        }
        .severity-badge {
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-left: 1rem;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
        }
        .recommendation-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.8rem;
            border-left: 4px solid #2E8B57;
            transition: transform 0.2s;
        }
        .recommendation-card:hover {
            transform: translateX(5px);
        }
        .lifestyle-card {
            background-color: #f0f8ff;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 0.8rem;
            border-left: 4px solid #4682b4;
            transition: transform 0.2s;
        }
        .lifestyle-card:hover {
            transform: translateX(5px);
        }
        .metric-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            border: 1px solid #f0f0f0;
        }
        .section-title {
            color: #2E8B57;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid rgba(46, 139, 87, 0.2);
        }
        .risk-indicator {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        .risk-indicator:hover {
            filter: brightness(0.95);
        }
        .emergency-alert {
            background: linear-gradient(135deg, #ff6b6b20, #ff878720);
            border: 1px solid #ff8787;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.8rem;
            font-weight: 600;
            border-radius: 10px;
            background: linear-gradient(135deg, #2E8B57, #3CB371);
            color: white;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(46, 139, 87, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title"><h1>🧠 MindPulse.ai Analytics</h1></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="emergency-alert">
        ⚠️ <strong>Professional Screening Tool</strong><br>
        Emergency Resources:<br>
        • <strong>Tele-Manas</strong>: 14416 (24/7 support)<br>
        • <strong>Vandrevala Foundation</strong>: +91 9999 666 555 (24/7 support)<br>
        • <strong>Fortis Stress Helpline</strong>: +91 8376 804 102 (24/7 support)<br>
        • <strong>KIRAN</strong>: 1800-599-0019 (Mental health support & rehabilitation)<br>
        • <strong>One Life</strong>: 78930-78930 (Suicide prevention & crisis support)<br>
    </div>
""", unsafe_allow_html=True)

    
    tab1, tab2, tab3 = st.tabs(["Physical Health Metrics", "PHQ-9 Assessment", "Results"])
    
    # Physical Metrics Tab
    with tab1:
        st.markdown('<h2 class="section-title">Physical Health Metrics</h2>', unsafe_allow_html=True)
        physio_data = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            physio_data["Heart Rate"] = st.number_input("Average Heart Rate (bpm)", 40, 200, 70)
            physio_data["HRV"] = st.number_input("Heart Rate Variability (ms)", 0, 200, 50)
            physio_data["Sleep"] = st.number_input("Sleep Duration (hours)", 0.0, 24.0, 7.0)
            
        with col2:
            physio_data["Sleep Quality"] = st.slider("Sleep Quality", 1, 10, 5)
            physio_data["Activity"] = st.number_input("Daily Steps", 0, 50000, 8000)
            physio_data["Stress"] = st.slider("Stress Level", 1, 10, 5)
    
    # PHQ-9 Assessment Tab
    with tab2:
        st.markdown('<h2 class="section-title">Depression Screening (PHQ-9)</h2>', unsafe_allow_html=True)
        phq9_responses = {}
        
        for i, question in enumerate(PHQ9_QUESTIONS):
            phq9_responses[i] = st.slider(
                question,
                min_value=0,
                max_value=3,
                value=0,
                help="0: Not at all, 1: Several days, 2: More than half the days, 3: Nearly every day"
            )
    
    # Results Tab
    with tab3:
        if st.button("Generate Assessment"):
            result = get_assessment(phq9_responses)
            
            # Clinical Summary Card
            st.markdown(f"""
                <div class="result-card">
                    <h2 class="section-title">Clinical Summary</h2>
                    <div class="status-badge" style='background-color: {"#ff6b6b" if result["is_depressed"] else "#28a745"}; color: white;'>
                        {"Depressed" if result["is_depressed"] else "Not Depressed"}
                    </div>
                    <span class="severity-badge">
                        Severity: {result["severity"]}
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Risk Level Card
            risk_colors = {
                "High": "#ff6b6b",
                "Medium": "#ffd93d",
                "Low": "#28a745"
            }
            st.markdown(f"""
                <div class="result-card">
                    <h2 class="section-title">Risk Assessment</h2>
                    <div class="risk-indicator" style='background-color: {risk_colors[result["risk_level"]]}20;'>
                        Risk Level: {result["risk_level"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Recommendations
                st.markdown("""
                    <div class="result-card">
                        <h2 class="section-title">Recommendations</h2>
                """, unsafe_allow_html=True)
                
                for rec in result["recommendations"]:
                    st.markdown(f"""
                        <div class="recommendation-card">
                            {rec}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Lifestyle Suggestions
                st.markdown("""
                    <div class="result-card">
                        <h2 class="section-title">Lifestyle Suggestions</h2>
                """, unsafe_allow_html=True)
                
                for suggestion in LIFESTYLE_SUGGESTIONS:
                    st.markdown(f"""
                        <div class="lifestyle-card">
                            {suggestion}
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Wellness Metrics
                st.markdown("""
                    <div class="metric-container">
                        <h2 class="section-title">Wellness Metrics</h2>
                    </div>
                """, unsafe_allow_html=True)
                
                analytics = WellnessAnalytics()
                st.plotly_chart(
                    analytics.generate_wellness_radar(physio_data),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()