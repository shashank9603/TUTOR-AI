import streamlit as st

def apply_custom_css():
    """Apply custom CSS for dark theme and UI enhancements."""
    st.markdown("""
    <style>
    /* Dark Theme */
    .main {
        background-color: #111111;
        color: #FFFFFF;
    }
    
    .stApp {
        background-color: #111111;
    }
    
    /* Make cards look like in the image */
    .card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #333333;
    }
    
    /* Style headings */
    h1, h2, h3 {
        color: #FFFFFF;
    }
    
    /* Style buttons to match the dark theme */
    .stButton>button {
        background-color: #2E2E2E;
        color: #FFFFFF;
        border: 1px solid #444444;
        border-radius: 5px;
    }
    
    .stButton>button:hover {
        background-color: #3E3E3E;
        border: 1px solid #555555;
    }
    
    /* Primary button style */
    .primary-btn {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1E1E1E;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #1E88E5;
    }
    
    /* Custom cards */
    .nav-card {
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 20px;
        background-color: #1E1E1E;
        margin-bottom: 15px;
    }
    
    /* Icons */
    .icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    /* Mastery level colors */
    .mastery-high {
        color: #4CAF50;
    }
    
    .mastery-medium {
        color: #FFC107;
    }
    
    .mastery-low {
        color: #F44336;
    }
    
    /* Form styling */
    .stForm {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333333;
    }
    
    /* Table styling */
    .stTable {
        background-color: #1E1E1E;
    }
    
    /* Dataframe styling */
    .dataframe {
        background-color: #1E1E1E !important;
    }
    
    .dataframe th {
        background-color: #2E2E2E !important;
        color: white !important;
    }
    
    .dataframe td {
        background-color: #1E1E1E !important;
        color: white !important;
    }
    
    /* Quiz styling */
    .quiz-option {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #2E2E2E;
        transition: background-color 0.3s;
    }
    
    .quiz-option:hover {
        background-color: #3E3E3E;
    }
    
    .quiz-selected {
        background-color: #1E88E5;
    }
    
    .quiz-correct {
        background-color: #4CAF50;
    }
    
    .quiz-incorrect {
        background-color: #F44336;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2E2E2E;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* Note editor */
    .stTextArea textarea {
        background-color: #2E2E2E;
        color: white;
        border: 1px solid #444444;
    }
    
    /* Search box */
    .stTextInput input {
        background-color: #2E2E2E;
        color: white;
        border: 1px solid #444444;
    }
    
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #2E2E2E;
        color: white;
    }
    
    /* Heatmap styles for quiz results */
    .score-excellent {
        background-color: #4CAF50;
        color: white;
        padding: 3px 7px;
        border-radius: 3px;
    }
    
    .score-good {
        background-color: #8BC34A;
        color: white;
        padding: 3px 7px;
        border-radius: 3px;
    }
    
    .score-average {
        background-color: #FFC107;
        color: black;
        padding: 3px 7px;
        border-radius: 3px;
    }
    
    .score-below-average {
        background-color: #FF9800;
        color: white;
        padding: 3px 7px;
        border-radius: 3px;
    }
    
    .score-poor {
        background-color: #F44336;
        color: white;
        padding: 3px 7px;
        border-radius: 3px;
    }
    
    /* Topic card styles */
    .topic-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #333333;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .topic-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite ease-in-out;
    }
    </style>
    """, unsafe_allow_html=True)

def card_html(title, content, icon=None, color="#1E88E5"):
    """Generate HTML for a card with optional icon."""
    icon_html = f'<div style="font-size: 2rem; margin-bottom: 10px; color: {color};">{icon}</div>' if icon else ''
    
    return f"""
    <div style="background-color: #1E1E1E; border-radius: 10px; padding: 20px; 
                margin-bottom: 20px; border: 1px solid #333333;">
        {icon_html}
        <h3 style="margin-top: 0;">{title}</h3>
        <div>{content}</div>
    </div>
    """

def mastery_badge(mastery_percentage):
    """Generate HTML for a mastery badge with appropriate color."""
    if mastery_percentage >= 80:
        color = "#4CAF50"  # Green
        level = "Expert"
    elif mastery_percentage >= 60:
        color = "#8BC34A"  # Light Green
        level = "Proficient"
    elif mastery_percentage >= 40:
        color = "#FFC107"  # Yellow
        level = "Intermediate"
    elif mastery_percentage >= 20:
        color = "#FF9800"  # Orange
        level = "Basic"
    else:
        color = "#F44336"  # Red
        level = "Novice"
    
    return f"""
    <div style="display: inline-block; background-color: {color}; color: white; 
                padding: 5px 10px; border-radius: 15px; font-size: 0.8rem;">
        {level} ({mastery_percentage:.1f}%)
    </div>
    """

def loading_spinner(text="Loading..."):
    """Show a custom loading spinner with text."""
    return f"""
    <div style="display: flex; align-items: center; margin: 20px 0;">
        <div class="loading-pulse" style="width: 20px; height: 20px; border-radius: 50%; 
                background-color: #1E88E5; margin-right: 10px;"></div>
        <div>{text}</div>
    </div>
    """

def get_icon_for_topic_type(topic_type):
    """Return an appropriate icon for a topic type."""
    if topic_type == "fundamental":
        return "🔵"  # Blue circle
    elif topic_type == "derived":
        return "🟢"  # Green circle
    elif topic_type == "technique":
        return "🟠"  # Orange circle
    elif topic_type == "question":
        return "🔴"  # Red circle
    else:
        return "⚪"  # White circle