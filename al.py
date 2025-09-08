import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
import textwrap
import re
import google.generativeai as genai
import io
import json
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="HSE Risk Analysis Dashboard",
    page_icon="https://www.hse.ie/favicon-32x32.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# HSE Risk Analysis Dashboard\nProfessional risk management and analysis platform."
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* --- General Styles --- */
    .main-header {
        background-color: #045A4D; /* HSE Green */
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        border-radius: 0 0 10px 10px;
    }
    .main-header img {
        height: 50px;
        margin-right: 20px;
    }
    .main-header-text {
        text-align: left;
    }
    .login-container .stButton > button {
        background-color: white; color: black; border: 1px solid #ced4da;
    }
    .login-container .stButton > button:hover {
        background-color: #f0f2f6; border-color: #045A4D; color: #045A4D;
    }
    .metric-container {
        background: white; padding: 1rem; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem 0;
    }
    .section-header {
        background: #f8f9fa; padding: 1rem; border-radius: 8px;
        margin: 1rem 0; border-left: 4px solid #28a745;
    }
    .alert-box {
        padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid #dee2e6;
    }
    .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
    .alert-warning { background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }
    .alert-danger { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
    .alert-info { background-color: #cce7ff; border-color: #b3d9ff; color: #004085; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    .stSelectbox > div > div { border-radius: 8px; }
    .footer {
        text-align: center; padding: 2rem; color: #6c757d;
        border-top: 1px solid #dee2e6; margin-top: 3rem;
    }
    
    /* --- NEW: CSS for Print to PDF --- */
    @media print {
        /* Hide unwanted elements */
        .main-header, .stSidebar, .stFileUploader, .stButton, .footer, .stExpander, .login-container {
            display: none !important;
        }

        /* Ensure main content takes up the full page */
        .main .block-container {
            max-width: 100% !important;
            padding: 1cm !important;
        }

        /* Prevent sections from being split across pages */
        .pdf-section {
            page-break-inside: avoid !important;
            margin-bottom: 2rem;
        }
        
        /* Force a page break before major new sections for clean layout */
        .pdf-section-break {
             page-break-before: always !important;
        }

        /* General styling for print */
        body {
            background-color: white !important;
        }
        .section-header h3 {
            font-size: 16pt !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Professional Constants ---
class AppConfig:
    """Application configuration constants"""
    APP_VERSION = "2.3.0" # Updated version for Print to PDF
    APP_NAME = "HSE Risk Analysis Dashboard"
    AUTHOR = "Healthcare Risk Management Team"
    LAST_UPDATED = "September 2025"
    
    # Colour schemes for consistent branding
    COLOURS = {
        'primary': '#045A4D',
        'secondary': '#28a745',
        'danger': '#dc3545',
        'warning': '#ffc107',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }
    
    RISK_COLOURS = {
        'High': '#dc3545',
        'Medium': '#ffc107', 
        'Low': '#28a745'
    }

# --- Enhanced Hospital Locations ---
HOSPITAL_LOCATIONS = {
    "UHK": {"name": "University Hospital Kerry", "lat": 52.268, "lon": -9.692, "region": "South"},
    "MRTH": {"name": "Midlands Regional Hospital Tullamore", "lat": 53.279, "lon": -7.493, "region": "Midlands"},
    "CGH": {"name": "Cavan General Hospital", "lat": 53.983, "lon": -7.366, "region": "Northeast"},
    "LCH": {"name": "Louth County Hospital", "lat": 54.005, "lon": -6.398, "region": "Northeast"},
    "STJ": {"name": "St James's Hospital", "lat": 53.337, "lon": -6.301, "region": "Dublin"},
    "MRHP": {"name": "Midlands Regional Hospital Portlaoise", "lat": 53.036, "lon": -7.301, "region": "Midlands"},
    "BGH": {"name": "Bantry General Hospital", "lat": 51.681, "lon": -9.455, "region": "Southwest"},
    "NGH": {"name": "Nenagh General Hospital", "lat": 52.863, "lon": -8.204, "region": "Midwest"},
    "TUH": {"name": "Tipperary University Hospital", "lat": 52.358, "lon": -7.711, "region": "Midwest"},
    "WGH": {"name": "Wexford General Hospital", "lat": 52.342, "lon": -6.475, "region": "Southeast"},
    "Sligo": {"name": "Sligo University Hospital", "lat": 54.2743, "lon": -8.4621, "region": "Northwest"},
    "LHK": {"name": "Letterkenny University Hospital", "lat": 54.949, "lon": -7.749, "region": "Northwest"},
    "MPRH": {"name": "Merlin Park University Hospital", "lat": 53.280, "lon": -9.006, "region": "West"}
}

MAIN_CATEGORIES = [
    "Infrastructure, Equipment & Maintenance",
    "Water Quality & Pressure", 
    "Governance, Communication & Procedures",
    "Procurement & Contractor Management",
    "Other"
]

# --- Professional Error Handling ---
class DashboardError(Exception):
    """Custom exception for dashboard-specific errors"""
    pass

class DataProcessingError(DashboardError):
    """Exception for data processing errors"""
    pass

class APIError(DashboardError):
    """Exception for API-related errors"""
    pass

# --- Enhanced Helper Functions ---
@st.cache_resource
def download_nltk_stopwords() -> None:
    """Downloads NLTK stopwords with professional error handling."""
    try:
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords already available")
    except LookupError:
        with st.spinner("Downloading required language resources..."):
            try:
                nltk.download('stopwords', quiet=True)
                logger.info("Successfully downloaded NLTK stopwords")
            except Exception as e:
                logger.error(f"Failed to download NLTK stopwords: {e}")
                st.error("Failed to download required resources. Please contact support.")
                raise

def show_professional_header() -> None:
    """Display professional header with branding"""
    logo_url = "https://www.hse.ie/image-library/hse-site-logo-2021.svg"
    st.markdown(f"""
    <div class="main-header">
        <img src="{logo_url}" alt="HSE Logo">
        <div class="main-header-text">
            <h1 style="margin: 0; font-size: 2.2rem;">{AppConfig.APP_NAME}</h1>
            <p style="margin: 0;">Professional Healthcare Risk Management & Analytics Platform</p>
        </div>
    </div>
    <div style="text-align: center; margin-top: -1.5rem; margin-bottom: 2rem;">
         <small>Version {AppConfig.APP_VERSION} | Last Updated: {AppConfig.LAST_UPDATED}</small>
    </div>
    """, unsafe_allow_html=True)

def show_loading_spinner(text: str = "Processing...") -> None:
    """Show consistent loading spinner"""
    return st.spinner(f"⚙️ {text}")

def show_success_message(message: str) -> None:
    """Display professional success message"""
    st.markdown(f"""
    <div class="alert-box alert-success">
        <strong>✅ Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def show_warning_message(message: str) -> None:
    """Display professional warning message"""
    st.markdown(f"""
    <div class="alert-box alert-warning">
        <strong>⚠️ Warning:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def show_error_message(message: str) -> None:
    """Display professional error message"""
    st.markdown(f"""
    <div class="alert-box alert-danger">
        <strong>❌ Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def show_info_message(message: str) -> None:
    """Display professional info message"""
    st.markdown(f"""
    <div class="alert-box alert-info">
        <strong>ℹ️ Info:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def validate_uploaded_file(uploaded_file) -> bool:
    """Validate uploaded file meets requirements"""
    if uploaded_file is None:
        return False
    
    # Check file size (max 10MB)
    if uploaded_file.size > 10 * 1024 * 1024:
        show_error_message("File size exceeds 10MB limit. Please upload a smaller file.")
        return False
    
    # Check file extension
    if not uploaded_file.name.lower().endswith('.xlsx'):
        show_error_message("Please upload an Excel file with .xlsx extension.")
        return False
    
    return True

def load_and_merge_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Enhanced data loading with comprehensive error handling and validation
    """
    if not validate_uploaded_file(uploaded_file):
        return None
    
    try:
        with show_loading_spinner("Loading and validating Excel file..."):
            start_time = time.time()
            
            # Read Excel file
            xls = pd.ExcelFile(uploaded_file)
            sheet_names = xls.sheet_names
            
            if not sheet_names:
                show_error_message("The uploaded Excel file contains no worksheets.")
                return None
            
            logger.info(f"Found {len(sheet_names)} worksheets: {sheet_names}")
            
            all_sheets_df = []
            processing_errors = []

            # Process each sheet
            for i, sheet_name in enumerate(sheet_names):
                try:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=2)
                    
                    # Data validation
                    if df_sheet.empty:
                        logger.warning(f"Sheet '{sheet_name}' is empty, skipping...")
                        continue
                    
                    # Clean the data
                    df_sheet.dropna(how='all', inplace=True)
                    
                    # Forward fill merged cells
                    columns_to_unmerge = df_sheet.columns[:11]
                    for col in columns_to_unmerge:
                        df_sheet[col] = df_sheet[col].ffill()
                    
                    # Add facility identifier
                    df_sheet['HSE Facility'] = sheet_name
                    
                    all_sheets_df.append(df_sheet)
                    logger.info(f"Successfully processed sheet '{sheet_name}' with {len(df_sheet)} rows")
                    
                except Exception as e:
                    error_msg = f"Error processing sheet '{sheet_name}': {str(e)}"
                    processing_errors.append(error_msg)
                    logger.error(error_msg)

            # Show processing results
            if processing_errors:
                show_warning_message(f"Some sheets had issues: {'; '.join(processing_errors)}")
            
            if not all_sheets_df:
                show_error_message("No valid data found in any worksheet.")
                return None

            # Merge all sheets
            merged_df = pd.concat(all_sheets_df, ignore_index=True)
            merged_df.columns = merged_df.columns.str.strip()

            # Data cleaning and standardisation
            if 'Risk Rating' in merged_df.columns:
                merged_df['Risk Rating'] = merged_df['Risk Rating'].astype(str).str.strip()
                rating_map = {'Hign': 'High', 'Med': 'Medium'}
                merged_df['Risk Rating'].replace(rating_map, inplace=True)

            if 'Risk Impact Category' in merged_df.columns:
                merged_df['Risk Impact Category'] = merged_df['Risk Impact Category'].astype(str).str.strip()
                # Standardise impact categories
                category_replacements = {
                    r'Loos of Trust / confidence|loss of Confidence': 'Loss of Confidence / Trust',
                    r'Harm to Perso.*': 'Harm to Person'
                }
                for pattern, replacement in category_replacements.items():
                    merged_df['Risk Impact Category'].replace(to_replace=pattern, value=replacement, regex=True, inplace=True)

            if 'Location of Risk Source' in merged_df.columns:
                merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].astype(str).str.strip()
                # Handle multiple locations separated by '/'
                merged_df = merged_df.assign(**{
                    'Location of Risk Source': merged_df['Location of Risk Source'].str.split('/')
                }).explode('Location of Risk Source')
                merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].str.strip()
                # Remove empty entries
                merged_df = merged_df[merged_df['Location of Risk Source'].str.strip() != '']
                merged_df.reset_index(drop=True, inplace=True)

            # Fill NaN values in key categorical columns to prevent errors
            logger.info("Filling missing values in key categorical columns.")
            categorical_cols_to_fill = [
                'Risk Rating', 'Risk Impact Category', 'Location of Risk Source', 'Topical Category'
            ]
            for col in categorical_cols_to_fill:
                if col in merged_df.columns:
                    merged_df[col].fillna('Unknown', inplace=True)

            processing_time = time.time() - start_time
            logger.info(f"Data processing completed in {processing_time:.2f} seconds")
            
            show_success_message(f"Successfully processed {len(merged_df)} risk records from {len(sheet_names)} facilities.")
            
            return merged_df
            
    except Exception as e:
        logger.error(f"Critical error in data loading: {e}")
        show_error_message(f"Failed to process the Excel file: {str(e)}")
        return None

# --- Enhanced AI Integration ---
@st.cache_data(show_spinner=False)
def assign_gemini_topics_batch(_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Enhanced AI topic classification with better error handling and retry logic
    """
    df = _df.copy()
    
    if 'Topical Category' not in df.columns:
        df['AI-Generated Topic'] = "Other"
        return df

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        logger.error(f"Gemini API configuration failed: {e}")
        show_error_message("AI service configuration failed. Using default categorisation.")
        df['AI-Generated Topic'] = "Other"
        return df
    
    # Get unique topics
    unique_topics = [
        topic for topic in df['Topical Category'].dropna().unique() 
        if isinstance(topic, str) and topic.strip()
    ]
    
    if not unique_topics:
        df['AI-Generated Topic'] = "Other"
        return df

    try:
        with show_loading_spinner(f"AI is analysing {len(unique_topics)} unique risk categories..."):
            categories_str = "\n".join(f"- {cat}" for cat in MAIN_CATEGORIES)
            unique_topics_json = json.dumps(unique_topics)

            prompt = f"""
            You are an expert healthcare risk classification system. Analyse the following risk categories and classify each into the most appropriate predefined category.
            
            **Predefined Categories:**
            {categories_str}
            
            **Classification Rules:**
            - Return ONLY a valid JSON object
            - Each key must be the exact original text from the input
            - Each value must be exactly one of the predefined categories
            - If unsure, use "Other"
            - Maintain consistency across similar terms
            
            **Risk Categories to Classify:**
            {unique_topics_json}
            
            **JSON Response:**
            """
            
            # Retry logic for API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    
                    # Extract JSON from response
                    cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if not cleaned_response:
                        raise ValueError("No valid JSON found in AI response")
                    
                    category_map = json.loads(cleaned_response.group(0))
                    
                    # Validate response
                    if not isinstance(category_map, dict):
                        raise ValueError("AI response is not a valid dictionary")
                    
                    # Map categories
                    df['AI-Generated Topic'] = df['Topical Category'].map(category_map).fillna("Other")
                    
                    # Log success
                    successful_mappings = len([v for v in category_map.values() if v != "Other"])
                    logger.info(f"AI successfully classified {successful_mappings}/{len(unique_topics)} categories")
                    
                    show_success_message(f"AI classification completed successfully! Classified {successful_mappings}/{len(unique_topics)} categories.")
                    
                    return df
                    
                except json.JSONDecodeError as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"JSON decode error on attempt {attempt + 1}, retrying...")
                        time.sleep(1)
                        continue
                    else:
                        raise
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"API error on attempt {attempt + 1}: {e}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        raise
                        
    except Exception as e:
        logger.error(f"AI classification failed after retries: {e}")
        show_warning_message("AI classification encountered issues. Using default categorisation.")
        df['AI-Generated Topic'] = "Other"
        return df

@st.cache_data(show_spinner=False)  
def get_hospital_locations_batch(_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Enhanced hospital location lookup with better error handling
    """
    df = _df.copy()
    
    # Map known hospitals first
    df['name'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('name'))
    df['lat'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lat'))
    df['lon'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lon'))
    df['region'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('region'))
    
    # Find unknown facilities
    unknown_facilities = [
        fac for fac in df['HSE Facility'].dropna().unique() 
        if isinstance(fac, str) and fac.strip() and fac not in HOSPITAL_LOCATIONS
    ]
    
    if not unknown_facilities:
        show_success_message("All hospital locations found in directory.")
        return df

    try:
        with show_loading_spinner(f"Looking up {len(unknown_facilities)} unknown facilities..."):
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
            
            unique_facilities_json = json.dumps(unknown_facilities)
            prompt = f"""
            You are a geographic information expert for Irish healthcare facilities. 
            
            **Task:** Identify full official names and coordinates for these Irish hospital abbreviations.
            
            **Required JSON Format:**
            {{
                "ABBREVIATION": {{
                    "name": "Full Official Hospital Name",
                    "lat": latitude_number,
                    "lon": longitude_number
                }},
                ...
            }}
            
            **Rules:**
            - Use exact coordinates for Ireland
            - Use official hospital names
            - If unknown, use null for all values
            - Return only valid JSON
            
            **Hospital Abbreviations:**
            {unique_facilities_json}
            
            **JSON Response:**
            """

            response = model.generate_content(prompt)
            cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
            
            if cleaned_response:
                location_data = json.loads(cleaned_response.group(0))
                
                # Update dataframe with new locations
                for fac, data in location_data.items():
                    if data and data.get('name'):
                        mask = df['HSE Facility'] == fac
                        df.loc[mask, 'name'] = data['name']
                        df.loc[mask, 'lat'] = data.get('lat')
                        df.loc[mask, 'lon'] = data.get('lon')
                
                #show_success_message(f"Successfully located {len(location_data)} facilities.")
            else:
                show_warning_message("Could not parse location data from AI response.")
                
    except Exception as e:
        logger.error(f"Hospital geolocation failed: {e}")
        show_warning_message("Some hospital locations could not be determined.")
    
    return df

# --- Enhanced Visualisation Functions ---
def create_professional_wordcloud(text_series: pd.Series, title: str) -> Optional[io.BytesIO]:
    """Generate professional word cloud with custom styling"""
    full_text = ' '.join(text_series.dropna().astype(str))
    if not full_text or len(full_text.strip()) < 10:
        return None
    
    try:
        stop_words_list = list(stopwords.words('english'))
        # Add custom healthcare stopwords
        healthcare_stopwords = ['hospital', 'patient', 'staff', 'department', 'service', 'care']
        stop_words_list.extend(healthcare_stopwords)
        
        wc = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            stopwords=stop_words_list,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(full_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return buf
        
    except Exception as e:
        logger.error(f"Word cloud generation failed: {e}")
        return None

def create_enhanced_metrics_display(df: pd.DataFrame) -> None:
    """Create professional metrics display with KPIs"""
    st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>📊 Risk Overview Dashboard</h3></div>', unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_risks = len(df)
        st.metric(
            label="Total Risks",
            value=f"{total_risks:,}",
            help="Total number of identified risks across all facilities"
        )
    
    with col2:
        facilities = df['HSE Facility'].nunique()
        st.metric(
            label="Facilities", 
            value=facilities,
            help="Number of HSE facilities included in analysis"
        )
    
    with col3:
        high_risks = len(df[df['Risk Rating'] == 'High'])
        high_risk_pct = (high_risks / total_risks * 100) if total_risks > 0 else 0
        st.metric(
            label="High Priority Risks",
            value=high_risks,
            delta=f"{high_risk_pct:.1f}% of total",
            delta_color="inverse",
            help="Number and percentage of high-priority risks requiring immediate attention"
        )
    
    with col4:
        categories = df['Risk Impact Category'].nunique()
        st.metric(
            label="Impact Categories",
            value=categories,
            help="Number of distinct risk impact categories identified"
        )
    st.markdown('</div>', unsafe_allow_html=True)

def create_professional_filters(df: pd.DataFrame) -> Dict:
    """Create professional sidebar filters with better UX"""
    st.sidebar.markdown("### 🎛️ Analysis Filters")
    st.sidebar.markdown("---")
    
    filters = {}
    
    # Facility filter with regions
    st.sidebar.markdown("**🏥 Healthcare Facilities**")
    facility_options = sorted(df['HSE Facility'].unique())
    filters['facilities'] = st.sidebar.multiselect(
        "Select facilities to analyse:",
        options=facility_options,
        default=facility_options,
        help="Choose specific HSE facilities to include in the analysis"
    )
    
    # Risk rating filter
    st.sidebar.markdown("**⚠️ Risk Priority Levels**")
    rating_options = sorted(df['Risk Rating'].unique())
    filters['risk_ratings'] = st.sidebar.multiselect(
        "Filter by risk priority:",
        options=rating_options,
        default=rating_options,
        help="Select risk priority levels to include"
    )
    
    # AI Topic filter
    st.sidebar.markdown("**🤖 AI-Generated Categories**")
    if 'AI-Generated Topic' in df.columns:
        ai_topic_options = sorted(df['AI-Generated Topic'].dropna().unique())
        filters['ai_topics'] = st.sidebar.multiselect(
            "Filter by AI classification:",
            options=ai_topic_options,
            default=ai_topic_options,
            help="AI-generated risk categories based on content analysis"
        )
    
    # Location filter
    st.sidebar.markdown("**📍 Risk Source Locations**")
    location_options = sorted(df['Location of Risk Source'].unique())
    filters['locations'] = st.sidebar.multiselect(
        "Filter by risk source:",
        options=location_options,
        default=location_options,
        help="Internal vs External risk sources"
    )
    
    # Impact category filter
    st.sidebar.markdown("**💥 Impact Categories**")
    impact_options = sorted(df['Risk Impact Category'].unique())
    filters['impact_categories'] = st.sidebar.multiselect(
        "Filter by impact type:",
        options=impact_options,
        default=impact_options,
        help="Types of potential impact from identified risks"
    )
    
    # Add clear filters button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset All Filters", help="Reset all filters to default values"):
        st.rerun()
    
    return filters

def map_topical_category(category) -> str:
    """Enhanced topical category mapping with better logic"""
    if pd.isna(category):
        return 'Other'
    
    category_lower = str(category).lower()
    
    # Define mapping rules with priority
    mapping_rules = [
        (['incoming water supply', 'water quality', 'supply pressure'], 'Supply Infrastructure & Quality'),
        (['distribution system', 'internal water', 'backup storage', 'water storage'], 'Distribution & Internal Systems'),
        (['metering', 'monitoring', 'measurement'], 'Metering & Monitoring'),
        (['protocol', 'eaps', 'sops', 'governance', 'procedure', 'communication'], 'Protocols & Governance'),
        (['maintenance', 'resources', 'procurement', 'contractor'], 'Maintenance & Resources'),
        (['documentation', 'data', 'drawings', 'maps', 'records'], 'Data & Documentation'),
        (['wastewater', 'stormwater', 'drainage'], 'Wastewater & Stormwater'),
        (['waste', 'conservation', 'non-potable'], 'Resource Management')
    ]
    
    for keywords, category_name in mapping_rules:
        if any(keyword in category_lower for keyword in keywords):
            return category_name
    
    return 'Other'

# --- Enhanced Dashboard Functions ---
def create_risk_distribution_analysis(df: pd.DataFrame) -> None:
    """Create comprehensive risk distribution analysis."""
    st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>📈 Risk Distribution Analysis</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 AI-Generated Topic Distribution")
        if 'AI-Generated Topic' in df.columns:
            ai_topic_counts = df['AI-Generated Topic'].value_counts()
            wrapped_labels = ['<br>'.join(textwrap.wrap(label, 25)) for label in ai_topic_counts.index]
            fig_ai_donut = px.pie(
                values=ai_topic_counts.values,
                names=wrapped_labels,
                hole=0.4,
                title="AI Classification Results",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_ai_donut.update_traces(textposition='inside', textinfo='percent+label')
            fig_ai_donut.update_layout(
                font=dict(size=12),
                showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
            )
            st.plotly_chart(fig_ai_donut, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Risk Priority Distribution")
        rating_counts = df['Risk Rating'].value_counts()
        fig_rating = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            labels={'x': 'Risk Priority Level', 'y': 'Number of Risks'},
            title="Risk Priority Breakdown",
            color=rating_counts.index,
            color_discrete_map=AppConfig.RISK_COLOURS
        )
        fig_rating.update_layout(showlegend=False, xaxis_title="Risk Priority Level", yaxis_title="Count")
        st.plotly_chart(fig_rating, use_container_width=True)

    # Additional analysis row
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("💥 Impact Category Analysis")
        impact_counts = df['Risk Impact Category'].value_counts()
        wrapped_labels_impact = ['<br>'.join(textwrap.wrap(label, 25)) for label in impact_counts.index]
        fig_impact = px.pie(
            values=impact_counts.values,
            names=wrapped_labels_impact,
            hole=0.3,
            title="Risk Impact Distribution"
        )
        fig_impact.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_impact, use_container_width=True)
    
    with col4:
        st.subheader("📍 Risk Source Location")
        location_counts = df['Location of Risk Source'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        fig_location = px.treemap(
            location_counts, 
            path=['Location'], 
            values='Count',
            title='Risk Source Distribution',
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_location, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def create_geographic_analysis(df: pd.DataFrame) -> None:
    """Create enhanced geographic risk analysis"""
    st.markdown('<div class="pdf-section pdf-section-break">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>🗺️ Geographic Risk Analysis</h3></div>', unsafe_allow_html=True)
    
    map_df = df.dropna(subset=['lat', 'lon']).copy()
    
    if map_df.empty:
        show_info_message("No geographic data available for mapping analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Risk Distribution Heatmap")
        
        # Create risk heatmap
        centre_lat, centre_lon = 53.4, -7.9
        m = folium.Map(
            location=[centre_lat, centre_lon], 
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add heatmap layer
        heat_data = [[row['lat'], row['lon']] for _, row in map_df.iterrows()]
        HeatMap(
            heat_data, 
            radius=25, 
            blur=15, 
            max_zoom=1, 
            min_opacity=0.2
        ).add_to(m)
        
        # Add facility markers with risk counts
        facility_risk_counts = map_df.groupby(['HSE Facility', 'name', 'lat', 'lon']).size().reset_index(name='risk_count')
        
        for _, row in facility_risk_counts.iterrows():
            if pd.notna(row['lat']) and pd.notna(row['lon']):
                popup_text = f"""
                <b>{row['name'] or row['HSE Facility']}</b><br>
                Total Risks: {row['risk_count']}<br>
                Facility Code: {row['HSE Facility']}
                """
                
                folium.CircleMarker(
                    location=[row['lat'], row['lon']],
                    radius=max(5, min(15, row['risk_count'] // 2)),
                    popup=folium.Popup(popup_text, max_width=300),
                    weight=0,
                    fill=True,
                    fillColor=AppConfig.COLOURS['primary'],
                    fillOpacity=0.7
                ).add_to(m)
        
        folium_static(m, height=500)
    
    with col2:
        # Top facilities by risk count
        st.subheader("Facilities by Risk Count")
        facility_summary = map_df.groupby(['HSE Facility', 'name']).size().reset_index(name='Total Risks')
        facility_summary = facility_summary.sort_values('Total Risks', ascending=False)
        
        # Use full name if available, otherwise facility code
        facility_summary['display_name'] = facility_summary['name'].fillna(facility_summary['HSE Facility'])

        facility_summary['display_name_wrapped'] = facility_summary['display_name'].apply(
            lambda x: '<br>'.join(textwrap.wrap(x, 20))
        )
        
        fig_donut = px.pie(
            facility_summary,
            names='display_name_wrapped',
            values='Total Risks',
            hole=0.4,
            title="Risk Distribution by Facility",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(facility_summary))
        fig_donut.update_layout(
            showlegend=False,
            font=dict(size=12),
            height=400,
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_advanced_analytics(df: pd.DataFrame) -> None:
    """Create advanced analytics section with a Sankey Diagram."""
    st.markdown('<div class="pdf-section pdf-section-break">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>🔬 Advanced Analytics</h3></div>', unsafe_allow_html=True)
    
    st.subheader("📊 Risk Flow Analysis (Sankey Diagram)")

    # Check for necessary columns
    required_cols = ['Location of Risk Source', 'Risk Rating', 'Parent Category']
    if not all(col in df.columns for col in required_cols):
        show_info_message("Insufficient data for the Risk Flow Analysis. Required columns are missing.")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    sankey_data = df.dropna(subset=required_cols)
    
    if sankey_data.empty:
        show_info_message("No data available to generate the Risk Flow Analysis chart.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
        
    all_nodes = list(pd.unique(sankey_data[required_cols].values.ravel('K')))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    palette = px.colors.qualitative.Plotly
    color_map = {node: palette[i % len(palette)] for i, node in enumerate(all_nodes)}
    
    links_1 = sankey_data.groupby(['Location of Risk Source', 'Risk Rating']).size().reset_index(name='value')
    links_1.columns = ['source_col', 'target_col', 'value']

    links_2 = sankey_data.groupby(['Risk Rating', 'Parent Category']).size().reset_index(name='value')
    links_2.columns = ['source_col', 'target_col', 'value']

    links = pd.concat([links_1, links_2], axis=0, ignore_index=True)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25, 
            thickness=20, 
            line=dict(color="black", width=0.5), 
            label=all_nodes, 
            color=[color_map[n] for n in all_nodes]
        ),
        link=dict(
            source=links['source_col'].map(node_map),
            target=links['target_col'].map(node_map),
            value=links['value'],
            color=[f"rgba({int(color_map[src][1:3], 16)}, {int(color_map[src][3:5], 16)}, {int(color_map[src][5:7], 16)}, 0.4)" for src in links['source_col']]
        )
    )])
    
    fig.update_layout(title_text="Risk Flow: Source → Priority → Category", font=dict(size=14), height=900)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def create_text_analytics(df: pd.DataFrame) -> None:
    """Create enhanced text analytics section"""
    st.markdown('<div class="pdf-section pdf-section-break">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>📝 Text Analytics & Insights</h3></div>', unsafe_allow_html=True)
    
    col1, spacer, col2 = st.columns([10, 1, 10]) # Use a spacer column
    
    with col1:
        st.subheader("☁️ Risk Description Word Cloud")
        if 'Risk Description' in df.columns:
            wc_img = create_professional_wordcloud(
                df['Risk Description'], 
                "Most Common Terms in Risk Descriptions"
            )
            if wc_img:
                st.image(wc_img, use_container_width=True)
            else:
                show_info_message("Insufficient text data for Risk Description word cloud.")
    
    with col2:
        st.subheader("💥 Impact Description Word Cloud") 
        if 'Impact Description' in df.columns:
            wc_img = create_professional_wordcloud(
                df['Impact Description'],
                "Most Common Terms in Impact Descriptions"
            )
            if wc_img:
                st.image(wc_img, use_container_width=True)
            else:
                show_info_message("Insufficient text data for Impact Description word cloud.")
    
    # Text statistics
    st.subheader("📊 Text Analytics Summary")
    
    text_metrics_cols = st.columns(4)
    
    if 'Risk Description' in df.columns:
        avg_risk_length = df['Risk Description'].astype(str).str.len().mean()
        with text_metrics_cols[0]:
            st.metric("Avg Risk Desc Length", f"{avg_risk_length:.0f} chars")
    
    if 'Impact Description' in df.columns:
        avg_impact_length = df['Impact Description'].astype(str).str.len().mean()
        with text_metrics_cols[1]:
            st.metric("Avg Impact Desc Length", f"{avg_impact_length:.0f} chars")
    
    if 'Topical Category' in df.columns:
        unique_categories = df['Topical Category'].nunique()
        with text_metrics_cols[2]:
            st.metric("Unique Categories", unique_categories)
    
    total_words = 0
    for col in ['Risk Description', 'Impact Description']:
        if col in df.columns:
            total_words += df[col].astype(str).str.split().str.len().sum()
    
    with text_metrics_cols[3]:
        st.metric("Total Words Analysed", f"{total_words:,}")
    st.markdown('</div>', unsafe_allow_html=True)

def create_executive_summary(df: pd.DataFrame) -> None:
    """Create executive summary section"""
    st.markdown('<div class="pdf-section pdf-section-break">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h3>📋 Executive Summary</h3></div>', unsafe_allow_html=True)
    
    # Key findings
    total_risks = len(df)
    high_priority = len(df[df['Risk Rating'] == 'High'])
    facilities_count = df['HSE Facility'].nunique()
    
    # Most common risk category
    if 'AI-Generated Topic' in df.columns:
        top_category = df['AI-Generated Topic'].mode().iloc[0] if not df['AI-Generated Topic'].empty else "Unknown"
        top_category_count = df['AI-Generated Topic'].value_counts().iloc[0] if not df['AI-Generated Topic'].empty else 0
    else:
        top_category = "Unknown"
        top_category_count = 0
    
    # Most affected facility
    top_facility = df['HSE Facility'].value_counts().index[0] if not df.empty else "Unknown"
    top_facility_risks = df['HSE Facility'].value_counts().iloc[0] if not df.empty else 0
    
    summary_text = f"""
    ## 🎯 Key Findings
    
    **Risk Overview:**
    - **{total_risks:,}** total risks identified across **{facilities_count}** HSE facilities
    - **{high_priority}** high-priority risks requiring immediate attention ({high_priority/total_risks*100:.1f}% of total)
    - **{top_category}** is the most common risk category with **{top_category_count}** incidents
    
    **Facility Impact:**
    - **{top_facility}** has the highest number of identified risks (**{top_facility_risks}** total)
    - Risk distribution varies significantly across facilities and regions
    
    **Recommendations:**
    1. **Immediate Action Required:** Focus on {high_priority} high-priority risks
    2. **Category Focus:** Prioritise improvements in {top_category} systems  
    3. **Facility Support:** Provide additional resources to {top_facility}
    4. **Systematic Review:** Implement standardised risk assessment protocols
    """
    
    st.markdown(summary_text)
    
    # Risk trend analysis (if date columns are available)
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower()]
    if date_columns:
        st.subheader("📈 Risk Trends")
        show_info_message("Trend analysis available - implement based on your specific date columns.")
    st.markdown('</div>', unsafe_allow_html=True)

def create_data_quality_report(df: pd.DataFrame) -> None:
    """Create data quality assessment"""
    st.markdown('<div class="section-header"><h3>🔍 Data Quality Report</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Completeness Analysis")
        
        completeness_data = []
        for col in df.columns:
            total_count = len(df)
            non_null_count = df[col].count()
            completeness_pct = (non_null_count / total_count * 100) if total_count > 0 else 0
            
            completeness_data.append({
                'Column': col,
                'Complete Records': non_null_count,
                'Total Records': total_count,
                'Completeness %': completeness_pct
            })
        
        completeness_df = pd.DataFrame(completeness_data)
        completeness_df = completeness_df.sort_values('Completeness %', ascending=False)
        
        # Colour code based on completeness
        def colour_completeness(val):
            if val >= 90:
                return 'background-color: #d4edda'  # Green
            elif val >= 70:
                return 'background-color: #fff3cd'  # Yellow  
            else:
                return 'background-color: #f8d7da'  # Red
        
        styled_df = completeness_df.style.applymap(
            colour_completeness, 
            subset=['Completeness %']
        ).format({'Completeness %': '{:.1f}%'})
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Data Quality Metrics")
        
        # Calculate quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        data_completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        
        # Standard deviation of completeness
        completeness_std = completeness_df['Completeness %'].std()
        
        st.metric("Overall Data Completeness", f"{data_completeness:.1f}%")
        st.metric("Duplicate Records", duplicate_rows)
        st.metric("Completeness Consistency", f"{100-completeness_std:.1f}%")
        
        # Quality score
        quality_score = (data_completeness + (100-completeness_std)) / 2
        quality_color = "normal"
        if quality_score >= 90:
            quality_color = "normal"
        elif quality_score >= 70:
            quality_color = "off"
        else:
            quality_color = "inverse"
            
        st.metric(
            "Data Quality Score", 
            f"{quality_score:.1f}%",
            delta_color=quality_color,
            help="Composite score based on completeness and consistency"
        )

def run_professional_dashboard():
    """Enhanced main dashboard function"""
    show_professional_header()
    
    # Download NLTK resources
    download_nltk_stopwords()
    
    # File upload section
    st.markdown("### 📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your HSE Risk Analysis Excel file",
        type="xlsx",
        help="Select an Excel file containing risk data across multiple facility worksheets"
    )
    
    if uploaded_file is None:
        show_info_message("Please upload an Excel file to begin the risk analysis.")
        
        # Show demo information
        with st.expander("ℹ️ About This Dashboard"):
            st.markdown("""
            This professional risk analysis dashboard provides:
            
            **🔧 Features:**
            - AI-powered risk categorisation using Google Gemini
            - Interactive geographic risk mapping
            - Advanced text analytics and word clouds  
            - Hierarchical risk flow analysis
            - Professional data quality reporting
            - Executive summary with key insights
            
            **📊 Analytics Capabilities:**
            - Risk distribution analysis across facilities
            - Priority-based risk assessment
            - Geographic hotspot identification
            - Text mining of risk descriptions
            
            **🚀 Getting Started:**
            1. Upload your Excel file with risk data
            2. Use the sidebar filters to focus your analysis
            3. Explore the interactive visualisations
            4. Review the executive summary for key insights
            """)
        return

    # Load and process data
    df = load_and_merge_data(uploaded_file)
    if df is None:
        return

    # Get API key
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        show_error_message("Gemini AI service is not properly configured. Please contact your administrator.")
        st.stop()

    # AI Processing
    with st.spinner("🤖 Processing data with AI..."):
        df = assign_gemini_topics_batch(df, api_key)
        df = get_hospital_locations_batch(df, api_key)

    # Add derived columns
    df['Parent Category'] = df['Topical Category'].apply(map_topical_category)

    # Professional filters
    filters = create_professional_filters(df)
    
    # --- NEW: PDF Export Instructions in Sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Export to PDF")
    st.sidebar.info(
        """
        1. Press `Ctrl+P` (or `Cmd+P`).
        2. Set **Destination** to **Save as PDF**.
        3. Set **Layout** to **Landscape**.
        4. Enable **Background graphics**.
        5. Click **Save**.
        """
    )

    # Apply filters
    df_filtered = df[
        df['HSE Facility'].isin(filters['facilities']) &
        df['Risk Rating'].isin(filters['risk_ratings']) &
        df['Location of Risk Source'].isin(filters['locations']) &
        df['Risk Impact Category'].isin(filters['impact_categories'])
    ].copy()
    
    if 'ai_topics' in filters:
        df_filtered = df_filtered[df_filtered['AI-Generated Topic'].isin(filters['ai_topics'])]
        
    if df_filtered.empty:
        show_warning_message("No data matches your current filter selection. Please adjust the filters.")
        return

    # Main dashboard sections
    create_enhanced_metrics_display(df_filtered)
    
    st.markdown("---")
    create_risk_distribution_analysis(df_filtered)
    
    st.markdown("---")
    create_advanced_analytics(df_filtered)
    
    st.markdown("---") 
    create_geographic_analysis(df_filtered)
    
    st.markdown("---")
    create_text_analytics(df_filtered)
    
    st.markdown("---")
    create_executive_summary(df_filtered)
    
    # Expandable sections
    with st.expander("🔍 Data Quality Assessment"):
        create_data_quality_report(df_filtered)
    
    with st.expander("📋 Filtered Dataset"):
        st.subheader("Complete Filtered Dataset")
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button for CSV
        csv_data = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name=f"hse_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    irish_tz = pytz.timezone("Europe/Dublin")
    irish_time = datetime.now(irish_tz)
    st.markdown("""
    <div class="footer">
        <hr>
        <p><strong>{}</strong> | Version {} | Generated on {}</p>
        <p>Created by Dave Maher | For HSE internal use.</p>
    </div>
    """.format(
        AppConfig.APP_NAME,
        AppConfig.APP_VERSION, 
        irish_time.strftime("%B %d, %Y at %I:%M %p")
    ), unsafe_allow_html=True)

def main():
    """Enhanced main function with professional authentication"""
    
    # Initialise session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    def authenticate_user():
        """Professional authentication with better UX"""
        try:
            if st.session_state.get("password") == st.secrets["PASSWORD"]:
                st.session_state.authenticated = True
                logger.info("User authenticated successfully")
                return True
            else:
                st.session_state.authenticated = False
                return False
        except KeyError:
            logger.error("Authentication password not configured in secrets")
            show_error_message("Authentication system not properly configured.")
            return False
    
    # Authentication UI
    if not st.session_state.authenticated:
        st.markdown("""
        <div class="main-header">
            <h1>🔐 Risk Analysis Dashboard</h1>
            <p>Secure Access Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Please enter your credentials to access the dashboard")
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input(
                "🔑 Access Password",
                type="password",
                key="password",
                help="Enter your authorised access password"
            )
            
            login_button = st.button(
                "Login",
                use_container_width=True
            )
            
            if login_button:
                if authenticate_user():
                    show_success_message("Authentication successful! Redirecting to dashboard...")
                    time.sleep(1)
                    st.rerun()
                else:
                    show_error_message("Invalid credentials. Please check your password and try again.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show system info
        st.markdown("---")
        st.info(f"""
        **System Information:**
        - Dashboard Version: {AppConfig.APP_VERSION}
        - Last Updated: {AppConfig.LAST_UPDATED}
        - Support: Contact your system administrator
        """)
        
        return

    # Main dashboard (authenticated users)
    if st.session_state.authenticated:
        
        # Add logo to sidebar
        logo_url = "https://tinteanhousing.eu/wp-content/uploads/2023/03/HSE-Logo.jpg"
        st.sidebar.markdown(
            f'<img src="{logo_url}" alt="HSE Logo" style="width: 100%; margin-bottom: 20px;">',
            unsafe_allow_html=True
        )

        
        # Sidebar authentication status
        st.sidebar.success("🟢 Authenticated")
        st.sidebar.markdown(f"**User:** Authorised Personnel")
        st.sidebar.markdown(f"**Session:** Active")
        
        if st.sidebar.button("Created by Dave Maher"):
            st.sidebar.write("This application intellectual property belongs to Dave Maher.")
        
        if st.sidebar.button("🚪 Logout", help="End your session and return to login"):
            st.session_state.authenticated = False
            logger.info("User logged out")
            st.rerun()
        
        st.sidebar.markdown("---")
        
        try:
            run_professional_dashboard()
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
            show_error_message(f"An unexpected error occurred: {str(e)}")
            st.stop()

if __name__ == '__main__':
    main()



# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from sklearn.feature_extraction.text import CountVectorizer
# import nltk
# from nltk.corpus import stopwords
# import numpy as np
# import textwrap
# import re
# import google.generativeai as genai
# import io
# import json
# import folium
# from streamlit_folium import folium_static
# from folium.plugins import HeatMap
# import sys
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import logging
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import time
# import pytz

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('dashboard.log'),
#         logging.StreamHandler(sys.stdout)
#     ]
# )
# logger = logging.getLogger(__name__)

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="HSE Risk Analysis Dashboard",
#     page_icon="https://www.hse.ie/favicon-32x32.png",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': None,
#         'Report a bug': None,
#         'About': "# HSE Risk Analysis Dashboard\nProfessional risk management and analysis platform."
#     }
# )

# # Custom CSS for professional styling
# st.markdown("""
# <style>
#     .main-header {
#         background-color: #045A4D; /* HSE Green */
#         display: flex;
#         align-items: center;
#         justify-content: center;
#         padding: 1.5rem 0;
#         margin: -1rem -1rem 2rem -1rem;
#         color: white;
#         text-align: center;
#         border-radius: 0 0 10px 10px;
#     }
#     .main-header img {
#         height: 50px;
#         margin-right: 20px;
#     }
#     .main-header-text {
#         text-align: left;
#     }
    
#     .login-container .stButton > button {
#         background-color: white;
#         color: black;
#         border: 1px solid #ced4da;
#     }
#     .login-container .stButton > button:hover {
#         background-color: #f0f2f6;
#         border-color: #045A4D;
#         color: #045A4D;
#     }
    
#     .metric-container {
#         background: white;
#         padding: 1rem;
#         border-radius: 10px;
#         box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         margin: 0.5rem 0;
#     }
    
#     .section-header {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border-left: 4px solid #28a745;
#     }
    
#     .alert-box {
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#         border: 1px solid #dee2e6;
#     }
    
#     .alert-success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
#     .alert-warning { background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }
#     .alert-danger { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
#     .alert-info { background-color: #cce7ff; border-color: #b3d9ff; color: #004085; }
    
#     .sidebar .sidebar-content {
#         background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
#     }
    
#     .stSelectbox > div > div {
#         border-radius: 8px;
#     }
    
#     .footer {
#         text-align: center;
#         padding: 2rem;
#         color: #6c757d;
#         border-top: 1px solid #dee2e6;
#         margin-top: 3rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- Professional Constants ---
# class AppConfig:
#     """Application configuration constants"""
#     APP_VERSION = "2.1.1"
#     APP_NAME = "HSE Risk Analysis Dashboard"
#     AUTHOR = "Healthcare Risk Management Team"
#     LAST_UPDATED = "September 2025"
    
#     # Colour schemes for consistent branding
#     COLOURS = {
#         'primary': '#045A4D',
#         'secondary': '#28a745',
#         'danger': '#dc3545',
#         'warning': '#ffc107',
#         'info': '#17a2b8',
#         'light': '#f8f9fa',
#         'dark': '#343a40'
#     }
    
#     RISK_COLOURS = {
#         'High': '#dc3545',
#         'Medium': '#ffc107', 
#         'Low': '#28a745'
#     }

# # --- Enhanced Hospital Locations ---
# HOSPITAL_LOCATIONS = {
#     "UHK": {"name": "University Hospital Kerry", "lat": 52.268, "lon": -9.692, "region": "South"},
#     "MRTH": {"name": "Midlands Regional Hospital Tullamore", "lat": 53.279, "lon": -7.493, "region": "Midlands"},
#     "CGH": {"name": "Cavan General Hospital", "lat": 53.983, "lon": -7.366, "region": "Northeast"},
#     "LCH": {"name": "Louth County Hospital", "lat": 54.005, "lon": -6.398, "region": "Northeast"},
#     "STJ": {"name": "St James's Hospital", "lat": 53.337, "lon": -6.301, "region": "Dublin"},
#     "MRHP": {"name": "Midlands Regional Hospital Portlaoise", "lat": 53.036, "lon": -7.301, "region": "Midlands"},
#     "BGH": {"name": "Bantry General Hospital", "lat": 51.681, "lon": -9.455, "region": "Southwest"},
#     "NGH": {"name": "Nenagh General Hospital", "lat": 52.863, "lon": -8.204, "region": "Midwest"},
#     "TUH": {"name": "Tipperary University Hospital", "lat": 52.358, "lon": -7.711, "region": "Midwest"},
#     "WGH": {"name": "Wexford General Hospital", "lat": 52.342, "lon": -6.475, "region": "Southeast"},
#     "Sligo": {"name": "Sligo University Hospital", "lat": 54.2743, "lon": -8.4621, "region": "Northwest"},
#     "LHK": {"name": "Letterkenny University Hospital", "lat": 54.949, "lon": -7.749, "region": "Northwest"},
#     "MPRH": {"name": "Merlin Park University Hospital", "lat": 53.280, "lon": -9.006, "region": "West"}
# }

# MAIN_CATEGORIES = [
#     "Infrastructure, Equipment & Maintenance",
#     "Water Quality & Pressure", 
#     "Governance, Communication & Procedures",
#     "Procurement & Contractor Management",
#     "Other"
# ]

# # --- Professional Error Handling ---
# class DashboardError(Exception):
#     """Custom exception for dashboard-specific errors"""
#     pass

# class DataProcessingError(DashboardError):
#     """Exception for data processing errors"""
#     pass

# class APIError(DashboardError):
#     """Exception for API-related errors"""
#     pass

# # --- Enhanced Helper Functions ---
# @st.cache_resource
# def download_nltk_stopwords() -> None:
#     """Downloads NLTK stopwords with professional error handling."""
#     try:
#         nltk.data.find('corpora/stopwords')
#         logger.info("NLTK stopwords already available")
#     except LookupError:
#         with st.spinner("Downloading required language resources..."):
#             try:
#                 nltk.download('stopwords', quiet=True)
#                 logger.info("Successfully downloaded NLTK stopwords")
#             except Exception as e:
#                 logger.error(f"Failed to download NLTK stopwords: {e}")
#                 st.error("Failed to download required resources. Please contact support.")
#                 raise

# def show_professional_header() -> None:
#     """Display professional header with branding"""
#     logo_url = "https://www.hse.ie/image-library/hse-site-logo-2021.svg"
#     st.markdown(f"""
#     <div class="main-header">
#         <img src="{logo_url}" alt="HSE Logo">
#         <div class="main-header-text">
#             <h1 style="margin: 0; font-size: 2.2rem;">{AppConfig.APP_NAME}</h1>
#             <p style="margin: 0;">Professional Healthcare Risk Management & Analytics Platform</p>
#         </div>
#     </div>
#     <div style="text-align: center; margin-top: -1.5rem; margin-bottom: 2rem;">
#          <small>Version {AppConfig.APP_VERSION} | Last Updated: {AppConfig.LAST_UPDATED}</small>
#     </div>
#     """, unsafe_allow_html=True)

# def show_loading_spinner(text: str = "Processing...") -> None:
#     """Show consistent loading spinner"""
#     return st.spinner(f"⚙️ {text}")

# def show_success_message(message: str) -> None:
#     """Display professional success message"""
#     st.markdown(f"""
#     <div class="alert-box alert-success">
#         <strong>✅ Success:</strong> {message}
#     </div>
#     """, unsafe_allow_html=True)

# def show_warning_message(message: str) -> None:
#     """Display professional warning message"""
#     st.markdown(f"""
#     <div class="alert-box alert-warning">
#         <strong>⚠️ Warning:</strong> {message}
#     </div>
#     """, unsafe_allow_html=True)

# def show_error_message(message: str) -> None:
#     """Display professional error message"""
#     st.markdown(f"""
#     <div class="alert-box alert-danger">
#         <strong>❌ Error:</strong> {message}
#     </div>
#     """, unsafe_allow_html=True)

# def show_info_message(message: str) -> None:
#     """Display professional info message"""
#     st.markdown(f"""
#     <div class="alert-box alert-info">
#         <strong>ℹ️ Info:</strong> {message}
#     </div>
#     """, unsafe_allow_html=True)

# def validate_uploaded_file(uploaded_file) -> bool:
#     """Validate uploaded file meets requirements"""
#     if uploaded_file is None:
#         return False
    
#     # Check file size (max 10MB)
#     if uploaded_file.size > 10 * 1024 * 1024:
#         show_error_message("File size exceeds 10MB limit. Please upload a smaller file.")
#         return False
    
#     # Check file extension
#     if not uploaded_file.name.lower().endswith('.xlsx'):
#         show_error_message("Please upload an Excel file with .xlsx extension.")
#         return False
    
#     return True

# def load_and_merge_data(uploaded_file) -> Optional[pd.DataFrame]:
#     """
#     Enhanced data loading with comprehensive error handling and validation
#     """
#     if not validate_uploaded_file(uploaded_file):
#         return None
    
#     try:
#         with show_loading_spinner("Loading and validating Excel file..."):
#             start_time = time.time()
            
#             # Read Excel file
#             xls = pd.ExcelFile(uploaded_file)
#             sheet_names = xls.sheet_names
            
#             if not sheet_names:
#                 show_error_message("The uploaded Excel file contains no worksheets.")
#                 return None
            
#             logger.info(f"Found {len(sheet_names)} worksheets: {sheet_names}")
            
#             all_sheets_df = []
#             processing_errors = []

#             # Process each sheet
#             for i, sheet_name in enumerate(sheet_names):
#                 try:
#                     df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=2)
                    
#                     # Data validation
#                     if df_sheet.empty:
#                         logger.warning(f"Sheet '{sheet_name}' is empty, skipping...")
#                         continue
                    
#                     # Clean the data
#                     df_sheet.dropna(how='all', inplace=True)
                    
#                     # Forward fill merged cells
#                     columns_to_unmerge = df_sheet.columns[:11]
#                     for col in columns_to_unmerge:
#                         df_sheet[col] = df_sheet[col].ffill()
                    
#                     # Add facility identifier
#                     df_sheet['HSE Facility'] = sheet_name
                    
#                     all_sheets_df.append(df_sheet)
#                     logger.info(f"Successfully processed sheet '{sheet_name}' with {len(df_sheet)} rows")
                    
#                 except Exception as e:
#                     error_msg = f"Error processing sheet '{sheet_name}': {str(e)}"
#                     processing_errors.append(error_msg)
#                     logger.error(error_msg)

#             # Show processing results
#             if processing_errors:
#                 show_warning_message(f"Some sheets had issues: {'; '.join(processing_errors)}")
            
#             if not all_sheets_df:
#                 show_error_message("No valid data found in any worksheet.")
#                 return None

#             # Merge all sheets
#             merged_df = pd.concat(all_sheets_df, ignore_index=True)
#             merged_df.columns = merged_df.columns.str.strip()

#             # Data cleaning and standardisation
#             if 'Risk Rating' in merged_df.columns:
#                 merged_df['Risk Rating'] = merged_df['Risk Rating'].astype(str).str.strip()
#                 rating_map = {'Hign': 'High', 'Med': 'Medium'}
#                 merged_df['Risk Rating'].replace(rating_map, inplace=True)

#             if 'Risk Impact Category' in merged_df.columns:
#                 merged_df['Risk Impact Category'] = merged_df['Risk Impact Category'].astype(str).str.strip()
#                 # Standardise impact categories
#                 category_replacements = {
#                     r'Loos of Trust / confidence|loss of Confidence': 'Loss of Confidence / Trust',
#                     r'Harm to Perso.*': 'Harm to Person'
#                 }
#                 for pattern, replacement in category_replacements.items():
#                     merged_df['Risk Impact Category'].replace(to_replace=pattern, value=replacement, regex=True, inplace=True)

#             if 'Location of Risk Source' in merged_df.columns:
#                 merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].astype(str).str.strip()
#                 # Handle multiple locations separated by '/'
#                 merged_df = merged_df.assign(**{
#                     'Location of Risk Source': merged_df['Location of Risk Source'].str.split('/')
#                 }).explode('Location of Risk Source')
#                 merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].str.strip()
#                 # Remove empty entries
#                 merged_df = merged_df[merged_df['Location of Risk Source'].str.strip() != '']
#                 merged_df.reset_index(drop=True, inplace=True)

#             # Fill NaN values in key categorical columns to prevent errors
#             logger.info("Filling missing values in key categorical columns.")
#             categorical_cols_to_fill = [
#                 'Risk Rating', 'Risk Impact Category', 'Location of Risk Source', 'Topical Category'
#             ]
#             for col in categorical_cols_to_fill:
#                 if col in merged_df.columns:
#                     merged_df[col].fillna('Unknown', inplace=True)

#             processing_time = time.time() - start_time
#             logger.info(f"Data processing completed in {processing_time:.2f} seconds")
            
#             show_success_message(f"Successfully processed {len(merged_df)} risk records from {len(sheet_names)} facilities.")
            
#             return merged_df
            
#     except Exception as e:
#         logger.error(f"Critical error in data loading: {e}")
#         show_error_message(f"Failed to process the Excel file: {str(e)}")
#         return None

# # --- Enhanced AI Integration ---
# @st.cache_data(show_spinner=False)
# def assign_gemini_topics_batch(_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
#     """
#     Enhanced AI topic classification with better error handling and retry logic
#     """
#     df = _df.copy()
    
#     if 'Topical Category' not in df.columns:
#         df['AI-Generated Topic'] = "Other"
#         return df

#     try:
#         # Configure Gemini
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
#     except Exception as e:
#         logger.error(f"Gemini API configuration failed: {e}")
#         show_error_message("AI service configuration failed. Using default categorisation.")
#         df['AI-Generated Topic'] = "Other"
#         return df
    
#     # Get unique topics
#     unique_topics = [
#         topic for topic in df['Topical Category'].dropna().unique() 
#         if isinstance(topic, str) and topic.strip()
#     ]
    
#     if not unique_topics:
#         df['AI-Generated Topic'] = "Other"
#         return df

#     try:
#         with show_loading_spinner(f"AI is analysing {len(unique_topics)} unique risk categories..."):
#             categories_str = "\n".join(f"- {cat}" for cat in MAIN_CATEGORIES)
#             unique_topics_json = json.dumps(unique_topics)

#             prompt = f"""
#             You are an expert healthcare risk classification system. Analyse the following risk categories and classify each into the most appropriate predefined category.
            
#             **Predefined Categories:**
#             {categories_str}
            
#             **Classification Rules:**
#             - Return ONLY a valid JSON object
#             - Each key must be the exact original text from the input
#             - Each value must be exactly one of the predefined categories
#             - If unsure, use "Other"
#             - Maintain consistency across similar terms
            
#             **Risk Categories to Classify:**
#             {unique_topics_json}
            
#             **JSON Response:**
#             """
            
#             # Retry logic for API calls
#             max_retries = 3
#             for attempt in range(max_retries):
#                 try:
#                     response = model.generate_content(prompt)
                    
#                     # Extract JSON from response
#                     cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
#                     if not cleaned_response:
#                         raise ValueError("No valid JSON found in AI response")
                    
#                     category_map = json.loads(cleaned_response.group(0))
                    
#                     # Validate response
#                     if not isinstance(category_map, dict):
#                         raise ValueError("AI response is not a valid dictionary")
                    
#                     # Map categories
#                     df['AI-Generated Topic'] = df['Topical Category'].map(category_map).fillna("Other")
                    
#                     # Log success
#                     successful_mappings = len([v for v in category_map.values() if v != "Other"])
#                     logger.info(f"AI successfully classified {successful_mappings}/{len(unique_topics)} categories")
                    
#                     show_success_message(f"AI classification completed successfully! Classified {successful_mappings}/{len(unique_topics)} categories.")
                    
#                     return df
                    
#                 except json.JSONDecodeError as e:
#                     if attempt < max_retries - 1:
#                         logger.warning(f"JSON decode error on attempt {attempt + 1}, retrying...")
#                         time.sleep(1)
#                         continue
#                     else:
#                         raise
#                 except Exception as e:
#                     if attempt < max_retries - 1:
#                         logger.warning(f"API error on attempt {attempt + 1}: {e}, retrying...")
#                         time.sleep(2)
#                         continue
#                     else:
#                         raise
                        
#     except Exception as e:
#         logger.error(f"AI classification failed after retries: {e}")
#         show_warning_message("AI classification encountered issues. Using default categorisation.")
#         df['AI-Generated Topic'] = "Other"
#         return df

# @st.cache_data(show_spinner=False)  
# def get_hospital_locations_batch(_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
#     """
#     Enhanced hospital location lookup with better error handling
#     """
#     df = _df.copy()
    
#     # Map known hospitals first
#     df['name'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('name'))
#     df['lat'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lat'))
#     df['lon'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lon'))
#     df['region'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('region'))
    
#     # Find unknown facilities
#     unknown_facilities = [
#         fac for fac in df['HSE Facility'].dropna().unique() 
#         if isinstance(fac, str) and fac.strip() and fac not in HOSPITAL_LOCATIONS
#     ]
    
#     if not unknown_facilities:
#         show_success_message("All hospital locations found in directory.")
#         return df

#     try:
#         with show_loading_spinner(f"Looking up {len(unknown_facilities)} unknown facilities..."):
#             genai.configure(api_key=api_key)
#             model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
            
#             unique_facilities_json = json.dumps(unknown_facilities)
#             prompt = f"""
#             You are a geographic information expert for Irish healthcare facilities. 
            
#             **Task:** Identify full official names and coordinates for these Irish hospital abbreviations.
            
#             **Required JSON Format:**
#             {{
#                 "ABBREVIATION": {{
#                     "name": "Full Official Hospital Name",
#                     "lat": latitude_number,
#                     "lon": longitude_number
#                 }},
#                 ...
#             }}
            
#             **Rules:**
#             - Use exact coordinates for Ireland
#             - Use official hospital names
#             - If unknown, use null for all values
#             - Return only valid JSON
            
#             **Hospital Abbreviations:**
#             {unique_facilities_json}
            
#             **JSON Response:**
#             """

#             response = model.generate_content(prompt)
#             cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
            
#             if cleaned_response:
#                 location_data = json.loads(cleaned_response.group(0))
                
#                 # Update dataframe with new locations
#                 for fac, data in location_data.items():
#                     if data and data.get('name'):
#                         mask = df['HSE Facility'] == fac
#                         df.loc[mask, 'name'] = data['name']
#                         df.loc[mask, 'lat'] = data.get('lat')
#                         df.loc[mask, 'lon'] = data.get('lon')
                
#                 #show_success_message(f"Successfully located {len(location_data)} facilities.")
#             else:
#                 show_warning_message("Could not parse location data from AI response.")
                
#     except Exception as e:
#         logger.error(f"Hospital geolocation failed: {e}")
#         show_warning_message("Some hospital locations could not be determined.")
    
#     return df

# # --- Enhanced Visualisation Functions ---
# def create_professional_wordcloud(text_series: pd.Series, title: str) -> Optional[io.BytesIO]:
#     """Generate professional word cloud with custom styling"""
#     full_text = ' '.join(text_series.dropna().astype(str))
#     if not full_text or len(full_text.strip()) < 10:
#         return None
    
#     try:
#         stop_words_list = list(stopwords.words('english'))
#         # Add custom healthcare stopwords
#         healthcare_stopwords = ['hospital', 'patient', 'staff', 'department', 'service', 'care']
#         stop_words_list.extend(healthcare_stopwords)
        
#         wc = WordCloud(
#             width=800, 
#             height=400, 
#             background_color='white',
#             colormap='viridis',
#             stopwords=stop_words_list,
#             max_words=100,
#             relative_scaling=0.5,
#             min_font_size=10
#         ).generate(full_text)
        
#         fig, ax = plt.subplots(figsize=(10, 5))
#         ax.imshow(wc, interpolation='bilinear')
#         ax.axis('off')
#         ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
#         buf = io.BytesIO()
#         fig.savefig(buf, format='png', bbox_inches='tight', dpi=150, pad_inches=0.1)
#         plt.close(fig)
#         buf.seek(0)
#         return buf
        
#     except Exception as e:
#         logger.error(f"Word cloud generation failed: {e}")
#         return None

# def create_enhanced_metrics_display(df: pd.DataFrame) -> None:
#     """Create professional metrics display with KPIs"""
#     st.markdown('<div class="section-header"><h3>📊 Risk Overview Dashboard</h3></div>', unsafe_allow_html=True)
    
#     # Main metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         total_risks = len(df)
#         st.metric(
#             label="Total Risks",
#             value=f"{total_risks:,}",
#             help="Total number of identified risks across all facilities"
#         )
    
#     with col2:
#         facilities = df['HSE Facility'].nunique()
#         st.metric(
#             label="Facilities", 
#             value=facilities,
#             help="Number of HSE facilities included in analysis"
#         )
    
#     with col3:
#         high_risks = len(df[df['Risk Rating'] == 'High'])
#         high_risk_pct = (high_risks / total_risks * 100) if total_risks > 0 else 0
#         st.metric(
#             label="High Priority Risks",
#             value=high_risks,
#             delta=f"{high_risk_pct:.1f}% of total",
#             delta_color="inverse",
#             help="Number and percentage of high-priority risks requiring immediate attention"
#         )
    
#     with col4:
#         categories = df['Risk Impact Category'].nunique()
#         st.metric(
#             label="Impact Categories",
#             value=categories,
#             help="Number of distinct risk impact categories identified"
#         )

# def create_professional_filters(df: pd.DataFrame) -> Dict:
#     """Create professional sidebar filters with better UX"""
#     st.sidebar.markdown("### 🎛️ Analysis Filters")
#     st.sidebar.markdown("---")
    
#     filters = {}
    
#     # Facility filter with regions
#     st.sidebar.markdown("**🏥 Healthcare Facilities**")
#     facility_options = sorted(df['HSE Facility'].unique())
#     filters['facilities'] = st.sidebar.multiselect(
#         "Select facilities to analyse:",
#         options=facility_options,
#         default=facility_options,
#         help="Choose specific HSE facilities to include in the analysis"
#     )
    
#     # Risk rating filter
#     st.sidebar.markdown("**⚠️ Risk Priority Levels**")
#     rating_options = sorted(df['Risk Rating'].unique())
#     filters['risk_ratings'] = st.sidebar.multiselect(
#         "Filter by risk priority:",
#         options=rating_options,
#         default=rating_options,
#         help="Select risk priority levels to include"
#     )
    
#     # AI Topic filter
#     st.sidebar.markdown("**🤖 AI-Generated Categories**")
#     if 'AI-Generated Topic' in df.columns:
#         ai_topic_options = sorted(df['AI-Generated Topic'].dropna().unique())
#         filters['ai_topics'] = st.sidebar.multiselect(
#             "Filter by AI classification:",
#             options=ai_topic_options,
#             default=ai_topic_options,
#             help="AI-generated risk categories based on content analysis"
#         )
    
#     # Location filter
#     st.sidebar.markdown("**📍 Risk Source Locations**")
#     location_options = sorted(df['Location of Risk Source'].unique())
#     filters['locations'] = st.sidebar.multiselect(
#         "Filter by risk source:",
#         options=location_options,
#         default=location_options,
#         help="Internal vs External risk sources"
#     )
    
#     # Impact category filter
#     st.sidebar.markdown("**💥 Impact Categories**")
#     impact_options = sorted(df['Risk Impact Category'].unique())
#     filters['impact_categories'] = st.sidebar.multiselect(
#         "Filter by impact type:",
#         options=impact_options,
#         default=impact_options,
#         help="Types of potential impact from identified risks"
#     )
    
#     # Add clear filters button
#     st.sidebar.markdown("---")
#     if st.sidebar.button("🔄 Reset All Filters", help="Reset all filters to default values"):
#         st.rerun()
    
#     return filters

# def map_topical_category(category) -> str:
#     """Enhanced topical category mapping with better logic"""
#     if pd.isna(category):
#         return 'Other'
    
#     category_lower = str(category).lower()
    
#     # Define mapping rules with priority
#     mapping_rules = [
#         (['incoming water supply', 'water quality', 'supply pressure'], 'Supply Infrastructure & Quality'),
#         (['distribution system', 'internal water', 'backup storage', 'water storage'], 'Distribution & Internal Systems'),
#         (['metering', 'monitoring', 'measurement'], 'Metering & Monitoring'),
#         (['protocol', 'eaps', 'sops', 'governance', 'procedure', 'communication'], 'Protocols & Governance'),
#         (['maintenance', 'resources', 'procurement', 'contractor'], 'Maintenance & Resources'),
#         (['documentation', 'data', 'drawings', 'maps', 'records'], 'Data & Documentation'),
#         (['wastewater', 'stormwater', 'drainage'], 'Wastewater & Stormwater'),
#         (['waste', 'conservation', 'non-potable'], 'Resource Management')
#     ]
    
#     for keywords, category_name in mapping_rules:
#         if any(keyword in category_lower for keyword in keywords):
#             return category_name
    
#     return 'Other'

# # --- Enhanced Dashboard Functions ---
# def create_risk_distribution_analysis(df: pd.DataFrame) -> None:
#     """Create comprehensive risk distribution analysis"""
#     st.markdown('<div class="section-header"><h3>📈 Risk Distribution Analysis</h3></div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("🤖 AI-Generated Topic Distribution")
#         if 'AI-Generated Topic' in df.columns:
#             ai_topic_counts = df['AI-Generated Topic'].value_counts()
            
#             # --- MODIFICATION: Wrap long labels for better readability in the pie chart ---
#             wrapped_labels = ['<br>'.join(textwrap.wrap(label, 25)) for label in ai_topic_counts.index]
            
#             fig_ai_donut = px.pie(
#                 values=ai_topic_counts.values,
#                 names=wrapped_labels, # Use wrapped labels
#                 hole=0.4,
#                 title="AI Classification Results",
#                 color_discrete_sequence=px.colors.qualitative.Set3
#             )
#             fig_ai_donut.update_traces(textposition='inside', textinfo='percent+label')
#             fig_ai_donut.update_layout(
#                 font=dict(size=12),
#                 showlegend=True,
#                 legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01)
#             )
#             st.plotly_chart(fig_ai_donut, use_container_width=True)
    
#     with col2:
#         st.subheader("⚠️ Risk Priority Distribution")
#         rating_counts = df['Risk Rating'].value_counts()
        
#         fig_rating = px.bar(
#             x=rating_counts.index,
#             y=rating_counts.values,
#             labels={'x': 'Risk Priority Level', 'y': 'Number of Risks'},
#             title="Risk Priority Breakdown",
#             color=rating_counts.index,
#             color_discrete_map=AppConfig.RISK_COLOURS
#         )
#         fig_rating.update_layout(showlegend=False, xaxis_title="Risk Priority Level", yaxis_title="Count")
#         st.plotly_chart(fig_rating, use_container_width=True)

#     # Additional analysis row
#     col3, col4 = st.columns(2)
    
#     with col3:
#         st.subheader("💥 Impact Category Analysis")
#         impact_counts = df['Risk Impact Category'].value_counts()
        
#         # --- MODIFICATION: Wrap long labels for better readability in the pie chart ---
#         wrapped_labels_impact = ['<br>'.join(textwrap.wrap(label, 25)) for label in impact_counts.index]
        
#         fig_impact = px.pie(
#             values=impact_counts.values,
#             names=wrapped_labels_impact, # Use wrapped labels
#             hole=0.3,
#             title="Risk Impact Distribution"
#         )
#         fig_impact.update_traces(textposition='inside', textinfo='percent+label')
#         st.plotly_chart(fig_impact, use_container_width=True)
    
#     with col4:
#         st.subheader("📍 Risk Source Location")
#         location_counts = df['Location of Risk Source'].value_counts().reset_index()
#         location_counts.columns = ['Location', 'Count']
        
#         fig_location = px.treemap(
#             location_counts, 
#             path=['Location'], 
#             values='Count',
#             title='Risk Source Distribution',
#             color='Count',
#             color_continuous_scale='Blues'
#         )
#         st.plotly_chart(fig_location, use_container_width=True)

# def create_geographic_analysis(df: pd.DataFrame) -> None:
#     """Create enhanced geographic risk analysis"""
#     st.markdown('<div class="section-header"><h3>🗺️ Geographic Risk Analysis</h3></div>', unsafe_allow_html=True)
    
#     map_df = df.dropna(subset=['lat', 'lon']).copy()
    
#     if map_df.empty:
#         show_info_message("No geographic data available for mapping analysis.")
#         return
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("Risk Distribution Heatmap")
        
#         # Create risk heatmap
#         centre_lat, centre_lon = 53.4, -7.9
#         m = folium.Map(
#             location=[centre_lat, centre_lon], 
#             zoom_start=7,
#             tiles='OpenStreetMap'
#         )
        
#         # Add heatmap layer with a blue-yellow gradient
#         heat_data = [[row['lat'], row['lon']] for _, row in map_df.iterrows()]
#         HeatMap(
#             heat_data, 
#             radius=25, 
#             blur=15, 
#             max_zoom=1, 
#             min_opacity=0.2
#         ).add_to(m)
        
#         # Add facility markers with risk counts, now in brand green
#         facility_risk_counts = map_df.groupby(['HSE Facility', 'name', 'lat', 'lon']).size().reset_index(name='risk_count')
        
#         for _, row in facility_risk_counts.iterrows():
#             if pd.notna(row['lat']) and pd.notna(row['lon']):
#                 popup_text = f"""
#                 <b>{row['name'] or row['HSE Facility']}</b><br>
#                 Total Risks: {row['risk_count']}<br>
#                 Facility Code: {row['HSE Facility']}
#                 """
                
#                 # --- MODIFICATION: Make circle markers filled with no border for a cleaner look ---
#                 folium.CircleMarker(
#                     location=[row['lat'], row['lon']],
#                     radius=max(5, min(15, row['risk_count'] // 2)),
#                     popup=folium.Popup(popup_text, max_width=300),
#                     weight=0,  # Set border weight to 0 to remove it
#                     fill=True,
#                     fillColor=AppConfig.COLOURS['primary'],
#                     fillOpacity=0 # Make the fill semi-transparent and visible
#                 ).add_to(m)
        
#         folium_static(m, height=500)
    
#     with col2:
#         # Top facilities by risk count
#         st.subheader("Facilities by Risk Count")
#         facility_summary = map_df.groupby(['HSE Facility', 'name']).size().reset_index(name='Total Risks')
#         facility_summary = facility_summary.sort_values('Total Risks', ascending=False)
        
#         # Use full name if available, otherwise facility code
#         facility_summary['display_name'] = facility_summary['name'].fillna(facility_summary['HSE Facility'])

#         # --- MODIFICATION: Wrap long labels for better readability in the pie chart ---
#         facility_summary['display_name_wrapped'] = facility_summary['display_name'].apply(
#             lambda x: '<br>'.join(textwrap.wrap(x, 20))
#         )
        
#         fig_donut = px.pie(
#             facility_summary,
#             names='display_name_wrapped', # Use wrapped labels column
#             values='Total Risks',
#             hole=0.4,
#             title="Risk Distribution by Facility",
#             color_discrete_sequence=px.colors.qualitative.Pastel
#         )
#         fig_donut.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05] * len(facility_summary))
#         fig_donut.update_layout(
#             showlegend=False,
#             font=dict(size=12),
#             height=400,
#             margin=dict(t=50, b=0, l=0, r=0)
#         )
#         st.plotly_chart(fig_donut, use_container_width=True)

# def create_advanced_analytics(df: pd.DataFrame) -> None:
#     """Create advanced analytics section with a Sankey Diagram."""
#     st.markdown('<div class="section-header"><h3>🔬 Advanced Analytics</h3></div>', unsafe_allow_html=True)
    
#     st.subheader("📊 Risk Flow Analysis (Sankey Diagram)")

#     # Check for necessary columns
#     required_cols = ['Location of Risk Source', 'Risk Rating', 'Parent Category']
#     if not all(col in df.columns for col in required_cols):
#         show_info_message("Insufficient data for the Risk Flow Analysis. Required columns are missing.")
#         return

#     sankey_data = df.dropna(subset=required_cols)
    
#     if sankey_data.empty:
#         show_info_message("No data available to generate the Risk Flow Analysis chart.")
#         return
        
#     all_nodes = list(pd.unique(sankey_data[required_cols].values.ravel('K')))
#     node_map = {node: i for i, node in enumerate(all_nodes)}
#     palette = px.colors.qualitative.Plotly
#     color_map = {node: palette[i % len(palette)] for i, node in enumerate(all_nodes)}
    
#     links_1 = sankey_data.groupby(['Location of Risk Source', 'Risk Rating']).size().reset_index(name='value')
#     links_1.columns = ['source_col', 'target_col', 'value']

#     links_2 = sankey_data.groupby(['Risk Rating', 'Parent Category']).size().reset_index(name='value')
#     links_2.columns = ['source_col', 'target_col', 'value']

#     links = pd.concat([links_1, links_2], axis=0, ignore_index=True)
    
#     fig = go.Figure(data=[go.Sankey(
#         node=dict(
#             pad=25, 
#             thickness=20, 
#             line=dict(color="black", width=0.5), 
#             label=all_nodes, 
#             color=[color_map[n] for n in all_nodes]
#         ),
#         link=dict(
#             source=links['source_col'].map(node_map),
#             target=links['target_col'].map(node_map),
#             value=links['value'],
#             color=[f"rgba({int(color_map[src][1:3], 16)}, {int(color_map[src][3:5], 16)}, {int(color_map[src][5:7], 16)}, 0.4)" for src in links['source_col']]
#         )
#     )])
    
#     fig.update_layout(title_text="Risk Flow: Source → Priority → Category", font=dict(size=14), height=900)
#     st.plotly_chart(fig, use_container_width=True)


# def create_text_analytics(df: pd.DataFrame) -> None:
#     """Create enhanced text analytics section"""
#     st.markdown('<div class="section-header"><h3>📝 Text Analytics & Insights</h3></div>', unsafe_allow_html=True)
    
#     col1, spacer, col2 = st.columns([10, 1, 10]) # Use a spacer column
    
#     with col1:
#         st.subheader("☁️ Risk Description Word Cloud")
#         if 'Risk Description' in df.columns:
#             wc_img = create_professional_wordcloud(
#                 df['Risk Description'], 
#                 "Most Common Terms in Risk Descriptions"
#             )
#             if wc_img:
#                 st.image(wc_img, use_container_width=True)
#             else:
#                 show_info_message("Insufficient text data for Risk Description word cloud.")
    
#     with col2:
#         st.subheader("💥 Impact Description Word Cloud") 
#         if 'Impact Description' in df.columns:
#             wc_img = create_professional_wordcloud(
#                 df['Impact Description'],
#                 "Most Common Terms in Impact Descriptions"
#             )
#             if wc_img:
#                 st.image(wc_img, use_container_width=True)
#             else:
#                 show_info_message("Insufficient text data for Impact Description word cloud.")
    
#     # Text statistics
#     st.subheader("📊 Text Analytics Summary")
    
#     text_metrics_cols = st.columns(4)
    
#     if 'Risk Description' in df.columns:
#         avg_risk_length = df['Risk Description'].astype(str).str.len().mean()
#         with text_metrics_cols[0]:
#             st.metric("Avg Risk Desc Length", f"{avg_risk_length:.0f} chars")
    
#     if 'Impact Description' in df.columns:
#         avg_impact_length = df['Impact Description'].astype(str).str.len().mean()
#         with text_metrics_cols[1]:
#             st.metric("Avg Impact Desc Length", f"{avg_impact_length:.0f} chars")
    
#     if 'Topical Category' in df.columns:
#         unique_categories = df['Topical Category'].nunique()
#         with text_metrics_cols[2]:
#             st.metric("Unique Categories", unique_categories)
    
#     total_words = 0
#     for col in ['Risk Description', 'Impact Description']:
#         if col in df.columns:
#             total_words += df[col].astype(str).str.split().str.len().sum()
    
#     with text_metrics_cols[3]:
#         st.metric("Total Words Analysed", f"{total_words:,}")

# def create_executive_summary(df: pd.DataFrame) -> None:
#     """Create executive summary section"""
#     st.markdown('<div class="section-header"><h3>📋 Executive Summary</h3></div>', unsafe_allow_html=True)
    
#     # Key findings
#     total_risks = len(df)
#     high_priority = len(df[df['Risk Rating'] == 'High'])
#     facilities_count = df['HSE Facility'].nunique()
    
#     # Most common risk category
#     if 'AI-Generated Topic' in df.columns:
#         top_category = df['AI-Generated Topic'].mode().iloc[0] if not df['AI-Generated Topic'].empty else "Unknown"
#         top_category_count = df['AI-Generated Topic'].value_counts().iloc[0] if not df['AI-Generated Topic'].empty else 0
#     else:
#         top_category = "Unknown"
#         top_category_count = 0
    
#     # Most affected facility
#     top_facility = df['HSE Facility'].value_counts().index[0] if not df.empty else "Unknown"
#     top_facility_risks = df['HSE Facility'].value_counts().iloc[0] if not df.empty else 0
    
#     summary_text = f"""
#     ## 🎯 Key Findings
    
#     **Risk Overview:**
#     - **{total_risks:,}** total risks identified across **{facilities_count}** HSE facilities
#     - **{high_priority}** high-priority risks requiring immediate attention ({high_priority/total_risks*100:.1f}% of total)
#     - **{top_category}** is the most common risk category with **{top_category_count}** incidents
    
#     **Facility Impact:**
#     - **{top_facility}** has the highest number of identified risks (**{top_facility_risks}** total)
#     - Risk distribution varies significantly across facilities and regions
    
#     **Recommendations:**
#     1. **Immediate Action Required:** Focus on {high_priority} high-priority risks
#     2. **Category Focus:** Prioritise improvements in {top_category} systems  
#     3. **Facility Support:** Provide additional resources to {top_facility}
#     4. **Systematic Review:** Implement standardised risk assessment protocols
#     """
    
#     st.markdown(summary_text)
    
#     # Risk trend analysis (if date columns are available)
#     date_columns = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower()]
#     if date_columns:
#         st.subheader("📈 Risk Trends")
#         show_info_message("Trend analysis available - implement based on your specific date columns.")

# def create_data_quality_report(df: pd.DataFrame) -> None:
#     """Create data quality assessment"""
#     st.markdown('<div class="section-header"><h3>🔍 Data Quality Report</h3></div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Completeness Analysis")
        
#         completeness_data = []
#         for col in df.columns:
#             total_count = len(df)
#             non_null_count = df[col].count()
#             completeness_pct = (non_null_count / total_count * 100) if total_count > 0 else 0
            
#             completeness_data.append({
#                 'Column': col,
#                 'Complete Records': non_null_count,
#                 'Total Records': total_count,
#                 'Completeness %': completeness_pct
#             })
        
#         completeness_df = pd.DataFrame(completeness_data)
#         completeness_df = completeness_df.sort_values('Completeness %', ascending=False)
        
#         # Colour code based on completeness
#         def colour_completeness(val):
#             if val >= 90:
#                 return 'background-color: #d4edda'  # Green
#             elif val >= 70:
#                 return 'background-color: #fff3cd'  # Yellow  
#             else:
#                 return 'background-color: #f8d7da'  # Red
        
#         styled_df = completeness_df.style.applymap(
#             colour_completeness, 
#             subset=['Completeness %']
#         ).format({'Completeness %': '{:.1f}%'})
        
#         st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
#     with col2:
#         st.subheader("Data Quality Metrics")
        
#         # Calculate quality metrics
#         total_cells = df.shape[0] * df.shape[1]
#         missing_cells = df.isnull().sum().sum()
#         data_completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0
        
#         # Duplicate analysis
#         duplicate_rows = df.duplicated().sum()
        
#         # Standard deviation of completeness
#         completeness_std = completeness_df['Completeness %'].std()
        
#         st.metric("Overall Data Completeness", f"{data_completeness:.1f}%")
#         st.metric("Duplicate Records", duplicate_rows)
#         st.metric("Completeness Consistency", f"{100-completeness_std:.1f}%")
        
#         # Quality score
#         quality_score = (data_completeness + (100-completeness_std)) / 2
#         quality_color = "normal"
#         if quality_score >= 90:
#             quality_color = "normal"
#         elif quality_score >= 70:
#             quality_color = "off"
#         else:
#             quality_color = "inverse"
            
#         st.metric(
#             "Data Quality Score", 
#             f"{quality_score:.1f}%",
#             delta_color=quality_color,
#             help="Composite score based on completeness and consistency"
#         )

# def run_professional_dashboard():
#     """Enhanced main dashboard function"""
#     show_professional_header()
    
#     # Download NLTK resources
#     download_nltk_stopwords()
    
#     # File upload section
#     st.markdown("### 📁 Data Upload")
#     uploaded_file = st.file_uploader(
#         "Upload your HSE Risk Analysis Excel file",
#         type="xlsx",
#         help="Select an Excel file containing risk data across multiple facility worksheets"
#     )
    
#     if uploaded_file is None:
#         show_info_message("Please upload an Excel file to begin the risk analysis.")
        
#         # Show demo information
#         with st.expander("ℹ️ About This Dashboard"):
#             st.markdown("""
#             This professional risk analysis dashboard provides:
            
#             **🔧 Features:**
#             - AI-powered risk categorisation using Google Gemini
#             - Interactive geographic risk mapping
#             - Advanced text analytics and word clouds  
#             - Hierarchical risk flow analysis
#             - Professional data quality reporting
#             - Executive summary with key insights
            
#             **📊 Analytics Capabilities:**
#             - Risk distribution analysis across facilities
#             - Priority-based risk assessment
#             - Geographic hotspot identification
#             - Text mining of risk descriptions
            
#             **🚀 Getting Started:**
#             1. Upload your Excel file with risk data
#             2. Use the sidebar filters to focus your analysis
#             3. Explore the interactive visualisations
#             4. Review the executive summary for key insights
#             """)
#         return

#     # Load and process data
#     df = load_and_merge_data(uploaded_file)
#     if df is None:
#         return

#     # Get API key
#     try:
#         api_key = st.secrets["GEMINI_API_KEY"]
#     except KeyError:
#         show_error_message("Gemini AI service is not properly configured. Please contact your administrator.")
#         st.stop()

#     # AI Processing
#     with st.spinner("🤖 Processing data with AI..."):
#         df = assign_gemini_topics_batch(df, api_key)
#         df = get_hospital_locations_batch(df, api_key)

#     # Add derived columns
#     df['Parent Category'] = df['Topical Category'].apply(map_topical_category)

#     # Professional filters
#     filters = create_professional_filters(df)
    
#     # Apply filters
#     df_filtered = df[
#         df['HSE Facility'].isin(filters['facilities']) &
#         df['Risk Rating'].isin(filters['risk_ratings']) &
#         df['Location of Risk Source'].isin(filters['locations']) &
#         df['Risk Impact Category'].isin(filters['impact_categories'])
#     ].copy()
    
#     if 'ai_topics' in filters:
#         df_filtered = df_filtered[df_filtered['AI-Generated Topic'].isin(filters['ai_topics'])]

#     if df_filtered.empty:
#         show_warning_message("No data matches your current filter selection. Please adjust the filters.")
#         return

#     # Main dashboard sections
#     create_enhanced_metrics_display(df_filtered)
    
#     st.markdown("---")
#     create_risk_distribution_analysis(df_filtered)
    
#     st.markdown("---")
#     create_advanced_analytics(df_filtered)
    
#     st.markdown("---") 
#     create_geographic_analysis(df_filtered)
    
#     st.markdown("---")
#     create_text_analytics(df_filtered)
    
#     st.markdown("---")
#     create_executive_summary(df_filtered)
    
#     # Expandable sections
#     with st.expander("🔍 Data Quality Assessment"):
#         create_data_quality_report(df_filtered)
    
#     with st.expander("📋 Filtered Dataset"):
#         st.subheader("Complete Filtered Dataset")
#         st.dataframe(
#             df_filtered,
#             use_container_width=True,
#             hide_index=True
#         )
        
#         # Download button
#         csv_data = df_filtered.to_csv(index=False)
#         st.download_button(
#             label="📥 Download Filtered Data (CSV)",
#             data=csv_data,
#             file_name=f"hse_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#             mime="text/csv"
#         )
    
#     # Footer
#     # Set timezone to Dublin
#     irish_tz = pytz.timezone("Europe/Dublin")
    
#     # Get current time in Dublin
#     irish_time = datetime.now(irish_tz)
#     st.markdown("""
#     <div class="footer">
#         <hr>
#         <p><strong>{}</strong> | Version {} | Generated on {}</p>
#         <p>Created by Dave Maher | For HSE internal use.</p>
#     </div>
#     """.format(
#         AppConfig.APP_NAME,
#         AppConfig.APP_VERSION, 
#         irish_time.strftime("%B %d, %Y at %I:%M %p"),
#         datetime.now().year
#     ), unsafe_allow_html=True)

# def main():
#     """Enhanced main function with professional authentication"""
    
#     # Initialise session state
#     if "authenticated" not in st.session_state:
#         st.session_state.authenticated = False
    
#     def authenticate_user():
#         """Professional authentication with better UX"""
#         try:
#             if st.session_state.get("password") == st.secrets["PASSWORD"]:
#                 st.session_state.authenticated = True
#                 logger.info("User authenticated successfully")
#                 return True
#             else:
#                 st.session_state.authenticated = False
#                 return False
#         except KeyError:
#             logger.error("Authentication password not configured in secrets")
#             show_error_message("Authentication system not properly configured.")
#             return False
    
#     # Authentication UI
#     if not st.session_state.authenticated:
#         st.markdown("""
#         <div class="main-header">
#             <h1>🔐 Risk Analysis Dashboard</h1>
#             <p>Secure Access Portal</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown("### Please enter your credentials to access the dashboard")
        
#         st.markdown('<div class="login-container">', unsafe_allow_html=True)
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             st.text_input(
#                 "🔑 Access Password",
#                 type="password",
#                 key="password",
#                 help="Enter your authorised access password"
#             )
            
#             login_button = st.button(
#                 "Login",
#                 use_container_width=True
#             )
            
#             if login_button:
#                 if authenticate_user():
#                     show_success_message("Authentication successful! Redirecting to dashboard...")
#                     time.sleep(1)
#                     st.rerun()
#                 else:
#                     show_error_message("Invalid credentials. Please check your password and try again.")
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Show system info
#         st.markdown("---")
#         st.info(f"""
#         **System Information:**
#         - Dashboard Version: {AppConfig.APP_VERSION}
#         - Last Updated: {AppConfig.LAST_UPDATED}
#         - Support: Contact your system administrator
#         """)
        
#         return

#     # Main dashboard (authenticated users)
#     if st.session_state.authenticated:
        
#         # Add logo to sidebar
#         logo_url = "https://tinteanhousing.eu/wp-content/uploads/2023/03/HSE-Logo.jpg"
#         st.sidebar.markdown(
#             f'<img src="{logo_url}" alt="HSE Logo" style="width: 100%; margin-bottom: 20px;">',
#             unsafe_allow_html=True
#         )


        
#         # Sidebar authentication status
#         st.sidebar.success("🟢 Authenticated")
#         st.sidebar.markdown(f"**User:** Authorised Personnel")
#         st.sidebar.markdown(f"**Session:** Active")
        
#         if st.sidebar.button("Created by Dave Maher"):
#             st.sidebar.write("This application intellectual property belongs to Dave Maher.")
        
#         if st.sidebar.button("🚪 Logout", help="End your session and return to login"):
#             st.session_state.authenticated = False
#             logger.info("User logged out")
#             st.rerun()
        
#         st.sidebar.markdown("---")
        
#         try:
#             run_professional_dashboard()
#         except Exception as e:
#             logger.error(f"Dashboard error: {e}")
#             show_error_message(f"An unexpected error occurred: {str(e)}")
#             st.stop()

# if __name__ == '__main__':
#     main()
