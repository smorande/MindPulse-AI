import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sqlite3
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import io
from fpdf import FPDF
import plotly.figure_factory as ff
import hashlib
import warnings
from typing import Dict, List, Any, Optional
warnings.filterwarnings('ignore')

# Load environment variables and initialize client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css",
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    suppress_callback_exceptions=True
)

# Constants
DB_PATH = 'mental_health.db'
MINIMUM_AGE = 18
MAX_DAILY_ASSESSMENTS = 3
CONFIDENCE_THRESHOLD = 0.7
PHQ9_RISK_THRESHOLD = 15
BIAS_CATEGORIES = ['age', 'gender', 'ethnicity', 'socioeconomic']

# Questionnaire definitions
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

class PDFReport:
    def __init__(self):
        self.pdf = FPDF()

    def generate_report(self, assessment_data, figures_data):
        self.pdf.add_page()
        
        # Add header
        self.pdf.set_font('Arial', 'B', 24)
        self.pdf.cell(0, 20, 'Mental Health Assessment Report', 0, 1, 'C')
        
        # Add summary
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Assessment Summary', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 12)
        self.pdf.multi_cell(0, 10, assessment_data['summary'])
        
        # Add risk level
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, f"Risk Level: {assessment_data['risk_level'].upper()}", 0, 1, 'L')
        
        # Add recommendations
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Recommendations', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 12)
        for rec in assessment_data['recommendations']:
            self.pdf.cell(0, 10, f"• {rec}", 0, 1, 'L')
        
        # Add support resources
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, 'Support Resources', 0, 1, 'L')
        self.pdf.set_font('Arial', '', 12)
        for resource in assessment_data.get('support_resources', []):
            self.pdf.cell(0, 10, f"• {resource}", 0, 1, 'L')
        
        output = io.BytesIO()
        self.pdf.output(output)
        return output.getvalue()

class EnhancedAnalytics:
    @staticmethod
    def generate_mood_chart(phq9_score, historical_data=None):
        if historical_data is None:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=10)
            scores = [phq9_score * np.random.uniform(0.8, 1.2) for _ in range(10)]
            historical_data = pd.DataFrame({'date': dates, 'score': scores})
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['score'],
            mode='lines+markers',
            name='PHQ-9 Score',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title={'text': 'Mood Trend Analysis', 'font': {'size': 24}},
            xaxis_title='Date',
            yaxis_title='PHQ-9 Score',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig

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
            r=[75] * len(categories),
            theta=categories,
            fill='toself',
            name='Optimal Range',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(0, 255, 0, 0.5)')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Current Metrics',
            line=dict(color='#1f77b4', width=3)
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
            showlegend=True,
            title={'text': 'Wellness Metrics Analysis', 'font': {'size': 24}},
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig

    @staticmethod
    def generate_sleep_analysis(physio_data):
        """Generate sleep analysis visualization"""
        sleep_duration = physio_data.get('Average Sleep Duration (hours)', 7)
        sleep_quality = physio_data.get('Sleep Quality (1-10)', 7)
        
        total_minutes = sleep_duration * 60
        stages = {
            'Deep Sleep': round(total_minutes * 0.25),
            'Light Sleep': round(total_minutes * 0.45),
            'REM': round(total_minutes * 0.25),
            'Awake': round(total_minutes * 0.05)
        }
        
        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=list(stages.keys()),
            values=list(stages.values()),
            hole=.3,
            textinfo='label+percent',
            marker=dict(colors=['#1f77b4', '#17becf', '#2ca02c', '#d62728'])
        ))
        
        fig.update_layout(
            title={
                'text': f'Sleep Analysis (Quality Score: {sleep_quality}/10)',
                'font': {'size': 24}
            },
            annotations=[dict(
                text=f'{sleep_duration}h<br>Total',
                x=0.5,
                y=0.5,
                font_size=20,
                showarrow=False
            )],
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig

class ResponsibleAI:
    def __init__(self):
        self.version = "1.0"
        
    def analyze_bias(self, assessment_result: Dict) -> Dict:
        bias_analysis = {
            "potential_biases": [],
            "confidence_score": 0.0,
            "mitigation_applied": []
        }
        
        text = ' '.join([
            str(assessment_result.get('summary', '')),
            ' '.join(assessment_result.get('recommendations', [])),
            str(assessment_result.get('follow_up', ''))
        ]).lower()
        
        # Analyze for potential biases
        for category in BIAS_CATEGORIES:
            bias_score = self._calculate_bias_score(text, category)
            if bias_score > 0.3:
                bias_analysis["potential_biases"].append({
                    "category": category,
                    "score": bias_score
                })
        
        # Calculate confidence score
        bias_analysis["confidence_score"] = max(0.0, 1.0 - len(bias_analysis["potential_biases"]) * 0.1)
        
        return bias_analysis
    
    def _calculate_bias_score(self, text: str, category: str) -> float:
        bias_terms = {
            'age': ['young', 'old', 'elderly', 'teenage'],
            'gender': ['male', 'female', 'man', 'woman'],
            'ethnicity': ['ethnic', 'racial', 'cultural'],
            'socioeconomic': ['poor', 'rich', 'wealthy', 'disadvantaged']
        }
        
        count = sum(1 for term in bias_terms[category] if term in text)
        return min(count * 0.2, 1.0)

class MentalHealthAssessment:
    def __init__(self):
        self.rai = ResponsibleAI()
    
    def get_assessment(self, phq9_data: Dict, physio_data: Dict, context: str) -> Dict:
        try:
            # Get initial assessment
            result = self._get_gpt4_assessment(phq9_data, physio_data, context)
            
            # Analyze for bias
            bias_analysis = self.rai.analyze_bias(result)
            
            # Add metadata
            result['assessment_metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'bias_analysis': bias_analysis,
                'confidence_score': bias_analysis['confidence_score']
            }
            
            return result
            
        except Exception as e:
            print(f"Error in assessment: {str(e)}")
            return self._get_fallback_assessment()
    
    def _get_gpt4_assessment(self, phq9_data: Dict, physio_data: Dict, context: str) -> Dict:
        system_prompt = """You are a mental health assessment specialist operating under strict ethical guidelines.
        Provide an unbiased, evidence-based assessment that is clear about limitations and focused on support and resources.
        
        Respond with valid JSON format containing:
        {
            "severity": "one of: Minimal, Mild, Moderate, Moderately Severe, Severe",
            "summary": "detailed assessment text",
            "recommendations": ["list of 3-5 specific recommendations"],
            "risk_level": "one of: low, medium, high",
            "follow_up": "follow up plan text",
            "lifestyle_suggestions": ["list of 3-4 suggestions"],
            "support_resources": ["list of 3-4 resources"]
        }"""

        user_prompt = f"""
        Based on:
        
        PHQ-9 Scores (0-3 scale):
        {json.dumps(phq9_data, indent=2)}
        
        Physical Health Metrics:
        {json.dumps(physio_data, indent=2)}
        
        Additional Context:
        {context}
        
        Provide an unbiased assessment in strict JSON format only.
        Include evidence-based recommendations and clear limitations.
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            result['phq9_score'] = sum(phq9_data.values())
            
            # Validate severity based on PHQ-9 score
            phq9_score = result['phq9_score']
            severity = "Minimal"
            if phq9_score >= 20:
                severity = "Severe"
            elif phq9_score >= 15:
                severity = "Moderately Severe"
            elif phq9_score >= 10:
                severity = "Moderate"
            elif phq9_score >= 5:
                severity = "Mild"
            result['severity'] = severity
            
            return result
            
        except Exception as e:
            print(f"Error in GPT-4 assessment: {str(e)}")
            raise
    
    def _get_fallback_assessment(self) -> Dict:
        """Provide fallback response"""
        return {
            "severity": "Unable to determine",
            "summary": "The assessment system is currently unavailable. This is an automated screening summary.",
            "recommendations": [
                "Consult with a mental health professional for a complete evaluation",
                "Continue monitoring your symptoms",
                "Maintain regular daily routines",
                "Stay connected with your support network"
            ],
            "risk_level": "unknown",
            "follow_up": "Please consult with a mental health professional for a thorough assessment.",
            "lifestyle_suggestions": [
                "Maintain regular physical activity",
                "Practice good sleep hygiene",
                "Engage in stress-reducing activities"
            ],
            "support_resources": [
                "National Crisis Hotline: 988",
                "SAMHSA National Helpline: 1-800-662-4357",
                "National Alliance on Mental Illness: 1-800-950-6264",
                "Crisis Text Line: Text HOME to 741741"
            ],
            "phq9_score": 0,
            "assessment_metadata": {
                "timestamp": datetime.now().isoformat(),
                "confidence_score": 0.0
            }
        }

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        phq9_score INTEGER,
        assessment_data TEXT,
        user_metrics TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

def create_question_card(question, q_id, q_type="phq9"):
    """Create a card for a single question"""
    card_content = [
        html.P(question, className="mb-2"),
        dcc.Slider(
            id=f'q-{q_id}',
            min=0,
            max=3,
            marks={
                0: {'label': 'Not at all', 'style': {'fontSize': '10px'}},
                1: {'label': 'Several days', 'style': {'fontSize': '10px'}},
                2: {'label': 'More than half', 'style': {'fontSize': '10px'}},
                3: {'label': 'Nearly daily', 'style': {'fontSize': '10px'}}
            },
            value=0,
            included=True
        ) if q_type == "phq9" else
        dbc.Input(
            id=f'p-{q_id}',
            type="number",
            placeholder="Enter value",
            className="form-control-sm"
        )
    ]
    
    return html.Div(
        dbc.Card(dbc.CardBody(card_content)),
        className="animate__animated animate__fadeIn mb-3"
    )

# App layout
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-brain mr-2"),
                "Mental Health Analytics"
            ], className="ms-2")
        ]),
        color="primary",
        dark=True,
        className="mb-4"
    ),
    
    dbc.Alert([
        html.I(className="fas fa-exclamation-triangle me-2"),
        html.Span("This is a screening tool only. Seek professional help for severe symptoms."),
        html.Div([
            html.Strong("Emergency Resources:"),
            html.Br(),
            "National Crisis Hotline: 988",
            html.Br(),
            "Emergency: 911"
        ], className="mt-2")
    ], color="info", dismissable=True, className="mb-4 animate__animated animate__fadeIn"),
    
    # Main Assessment Tabs
    dbc.Tabs([
        dbc.Tab([
            html.H5("PHQ-9 Assessment", className="text-center mb-4"),
            *[create_question_card(q, i) for i, q in enumerate(PHQ9_QUESTIONS)]
        ], label="Assessment", tab_id="tab-1"),
        
        dbc.Tab([
            html.H5("Physical Metrics", className="text-center mb-4"),
            *[create_question_card(m, i, "physio") for i, m in enumerate(PHYSIO_METRICS)]
        ], label="Metrics", tab_id="tab-2"),
        
        dbc.Tab([
            html.H5("Additional Context", className="text-center mb-4"),
            dbc.Textarea(
                id="context-input",
                placeholder="Share any additional thoughts...",
                style={"height": "150px"},
                className="mb-4"
            )
        ], label="Notes", tab_id="tab-3")
    ], id="tabs", active_tab="tab-1", className="mb-4"),
    
    # Submit Button
    dbc.Button([
        html.I(className="fas fa-chart-line me-2"),
        "Generate Assessment"
    ],
    id="submit-button",
    color="primary",
    size="lg",
    className="mb-4 w-100 animate__animated animate__pulse"),
    
    # Loading spinner and output
    dbc.Spinner(html.Div(id="assessment-output")),
    
    # Download button
    html.Div(
        dbc.Button([
            html.I(className="fas fa-download me-2"),
            "Download Report"
        ],
        id="download-button",
        color="success",
        className="mt-4 w-100",
        style={'display': 'none'}),
        id="download-button-container"
    ),
    
    # Hidden components
    dcc.Download(id="download-report"),
    dcc.Store(id="assessment-store"),
    dcc.Store(id="figures-store"),
    
    # Footer
    html.Footer([
        html.Hr(),
        html.P([
            "© 2024 Mental Health Analytics - For screening purposes only",
            html.Br(),
            html.Small("Version 1.0 - Powered by Responsible AI")
        ], className="text-center text-muted")
    ], className="mt-4")
], fluid=True, className="p-4")

def create_assessment_output(result, figures):
    """Create the assessment output display"""
    risk_colors = {
        "low": "success",
        "medium": "warning",
        "high": "danger",
        "unknown": "secondary"
    }
    
    risk_color = risk_colors.get(result['risk_level'], "secondary")
    confidence_score = result.get('assessment_metadata', {}).get('confidence_score', 0)
    
    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.H4(
                    f"Assessment Results - {result['severity']} Depression",
                    className="text-center animate__animated animate__fadeIn"
                )
            ]),
            dbc.CardBody([
                # Confidence Score Alert
                dbc.Alert([
                    html.I(className="fas fa-info-circle me-2"),
                    f"Confidence Score: {confidence_score:.2%}"
                ], 
                color="warning" if confidence_score < CONFIDENCE_THRESHOLD else "info",
                className="mb-3"),
                
                dbc.Row([
                    # Left Column - Text Content
                    dbc.Col([
                        html.Div([
                            html.H5("Clinical Summary", className="mb-3"),
                            html.P(result['summary'], className="lead"),
                            
                            html.H5("Recommendations", className="mt-4 mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem(
                                    rec,
                                    className="animate__animated animate__fadeInLeft",
                                    style={"animation-delay": f"{i*0.1}s"}
                                ) for i, rec in enumerate(result['recommendations'])
                            ], className="mb-4"),
                            
                            dbc.Alert([
                                html.H5("Risk Level", className="alert-heading"),
                                html.P(f"Current risk level: {result['risk_level'].upper()}")
                            ], color=risk_color, className="mb-4 animate__animated animate__pulse"),
                            
                            html.H5("Lifestyle Suggestions", className="mt-4 mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem(
                                    suggestion,
                                    className="animate__animated animate__fadeInRight",
                                    style={"animation-delay": f"{i*0.1}s"}
                                ) for i, suggestion in enumerate(result.get('lifestyle_suggestions', []))
                            ], className="mb-4"),
                            
                            html.H5("Support Resources", className="mt-4 mb-3"),
                            dbc.ListGroup([
                                dbc.ListGroupItem(
                                    resource,
                                    className="animate__animated animate__fadeInRight",
                                    style={"animation-delay": f"{i*0.1}s"}
                                ) for i, resource in enumerate(result.get('support_resources', []))
                            ])
                        ], className="mb-4")
                    ], md=6),
                    
                    # Right Column - Visualizations
                    dbc.Col([
                        html.Div([
                            html.H5("Mood Trend Analysis", className="mb-3"),
                            dcc.Graph(
                                figure=figures['mood_trend'],
                                config={'responsive': True},
                                className="mb-4"
                            ),
                            
                            html.H5("Wellness Metrics", className="mt-4 mb-3"),
                            dcc.Graph(
                                figure=figures['wellness_radar'],
                                config={'responsive': True},
                                className="mb-4"
                            ),
                            
                            html.H5("Sleep Analysis", className="mt-4 mb-3"),
                            dcc.Graph(
                                figure=figures['sleep_analysis'],
                                config={'responsive': True}
                            )
                        ], className="animate__animated animate__fadeIn")
                    ], md=6)
                ])
            ])
        ], className="mt-4")
    ])

# Callbacks
@app.callback(
    [Output("assessment-output", "children"),
     Output("assessment-store", "data"),
     Output("figures-store", "data"),
     Output("download-button-container", "style")],
    Input("submit-button", "n_clicks"),
    [State(f"q-{i}", "value") for i in range(len(PHQ9_QUESTIONS))] +
    [State(f"p-{i}", "value") for i in range(len(PHYSIO_METRICS))] +
    [State("context-input", "value")],
    prevent_initial_call=True
)
def update_assessment(n_clicks, *values):
    if n_clicks is None:
        raise PreventUpdate
    
    phq9_values = values[:len(PHQ9_QUESTIONS)]
    physio_values = values[len(PHQ9_QUESTIONS):-1]
    context = values[-1] or ""
    
    if None in phq9_values or None in physio_values:
        return dbc.Alert(
            "Please complete all questions before generating the assessment.",
            color="warning",
            className="mt-3"
        ), None, None, {'display': 'none'}
    
    phq9_data = {str(i): val for i, val in enumerate(phq9_values)}
    physio_data = {metric: val for metric, val in zip(PHYSIO_METRICS, physio_values)}
    
    assessment_system = MentalHealthAssessment()
    result = assessment_system.get_assessment(phq9_data, physio_data, context)
    
    analytics = EnhancedAnalytics()
    figures = {
        'mood_trend': analytics.generate_mood_chart(result['phq9_score']),
        'wellness_radar': analytics.generate_wellness_radar(physio_data),
        'sleep_analysis': analytics.generate_sleep_analysis(physio_data)
    }
    
    return create_assessment_output(result, figures), result, figures, {'display': 'block'}

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)