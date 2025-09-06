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
from folium.plugins import HeatMap
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import logging
from datetime import datetime
import time

# --- Logging Configuration ---
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
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# HSE Risk Analysis Dashboard\nProfessional risk management and analysis platform."
    }
)

# --- Professional Styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e5f8a 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
        border-radius: 0 0 10px 10px;
    }
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1f4e79;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Professional Constants ---
class AppConfig:
    APP_VERSION = "2.0.0"
    APP_NAME = "HSE Risk Analysis Dashboard"

# --- Predefined Hospital Locations ---
HOSPITAL_LOCATIONS = {
    "UHK": {"name": "University Hospital Kerry", "lat": 52.268, "lon": -9.692},
    "MRTH": {"name": "Midlands Regional Hospital Tullamore", "lat": 53.279, "lon": -7.493},
    "CGH": {"name": "Cavan General Hospital", "lat": 53.983, "lon": -7.366},
    "LCH": {"name": "Louth County Hospital", "lat": 54.005, "lon": -6.398},
    "STJ": {"name": "St James's Hospital", "lat": 53.337, "lon": -6.301},
    "MRHP": {"name": "Midlands Regional Hospital Portlaoise", "lat": 53.036, "lon": -7.301},
    "BGH": {"name": "Bantry General Hospital", "lat": 51.681, "lon": -9.455},
    "NGH": {"name": "Nenagh General Hospital", "lat": 52.863, "lon": -8.204},
    "TUH": {"name": "Tipperary University Hospital", "lat": 52.358, "lon": -7.711},
    "WGH": {"name": "Wexford General Hospital", "lat": 52.342, "lon": -6.475},
    "Sligo": {"name": "Sligo University Hospital", "lat": 54.2743, "lon": -8.4621},
    "LHK": {"name": "Letterkenny University Hospital", "lat": 54.949, "lon": -7.749},
    "MPRH": {"name": "Merlin Park University Hospital", "lat": 53.280, "lon": -9.006}
}

# --- Gemini AI Categories ---
MAIN_CATEGORIES = [
    "Infrastructure, Equipment & Maintenance",
    "Water Quality & Pressure",
    "Governance, Communication & Procedures",
    "Procurement & Contractor Management",
    "Other"
]

# --- Helper Functions ---
@st.cache_resource
def download_nltk_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def load_and_merge_data(uploaded_file):
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        all_sheets_df = []
        for sheet_name in sheet_names:
            try:
                df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=2)
                df_sheet.dropna(how='all', inplace=True)
                columns_to_unmerge = df_sheet.columns[:11]
                for col in columns_to_unmerge:
                    df_sheet[col] = df_sheet[col].ffill()
                df_sheet['HSE Facility'] = sheet_name
                all_sheets_df.append(df_sheet)
            except Exception as e:
                st.warning(f"Could not process sheet: '{sheet_name}'. Error: {e}")
        
        if not all_sheets_df:
            st.error("No data could be processed from the uploaded file.")
            return None

        merged_df = pd.concat(all_sheets_df, ignore_index=True)
        merged_df.columns = merged_df.columns.str.strip()

        if 'Risk Rating' in merged_df.columns:
            merged_df['Risk Rating'] = merged_df['Risk Rating'].str.strip()
            merged_df['Risk Rating'].replace({'Hign': 'High', 'Med': 'Medium'}, inplace=True)

        if 'Risk Impact Category' in merged_df.columns:
            merged_df['Risk Impact Category'] = merged_df['Risk Impact Category'].str.strip()
            merged_df['Risk Impact Category'].replace(to_replace=r'Loos of Trust / confidence|loss of Confidence', value='Loss of Confidence / Trust', regex=True, inplace=True)
            merged_df['Risk Impact Category'].replace(to_replace=r'Harm to Perso.*', value='Harm to Person', regex=True, inplace=True)

        if 'Location of Risk Source' in merged_df.columns:
            merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].astype(str).str.strip()
            merged_df = merged_df.assign(**{'Location of Risk Source': merged_df['Location of Risk Source'].str.split('/')}).explode('Location of Risk Source')
            merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].str.strip()
            merged_df = merged_df[~merged_df['Location of Risk Source'].isin(['', 'nan'])]
            merged_df.reset_index(drop=True, inplace=True)

        return merged_df
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

@st.cache_data
def assign_gemini_topics_batch(_df, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"API Key Configuration Error: {e}")
        st.stop()
    
    df = _df.copy()
    if 'Topical Category' not in df.columns:
        return df

    unique_topics = [t for t in df['Topical Category'].dropna().unique() if isinstance(t, str) and t.strip()]
    if not unique_topics:
        df['AI-Generated Topic'] = "Other"
        return df

    prompt = f"""
    You are an expert data classification system. Your task is to classify a list of topical issues into a set of predefined categories.
    Your response MUST be a single, valid JSON object where each key is the original issue text from the input list, and the corresponding value is ONLY the name of the most appropriate category from the predefined list.
    **Predefined Categories:**
    - {"\n- ".join(MAIN_CATEGORIES)}
    **JSON List of Issues to Classify:**
    {json.dumps(unique_topics)}
    **Your JSON Response:**
    """
    
    with st.spinner(f"AI is classifying {len(unique_topics)} unique topics..."):
        try:
            response = model.generate_content(prompt)
            cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not cleaned_response: raise ValueError("AI response did not contain a valid JSON object.")
            category_map = json.loads(cleaned_response.group(0))
            df['AI-Generated Topic'] = df['Topical Category'].map(category_map).fillna("Other")
        except Exception as e:
            st.error(f"AI topic classification failed: {e}")
            df['AI-Generated Topic'] = "Other"
    return df

@st.cache_data
def get_hospital_locations_batch(_df, api_key):
    df = _df.copy()
    df['name'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('name'))
    df['lat'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lat'))
    df['lon'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lon'))
    
    unknown_facilities = [fac for fac in df['HSE Facility'].dropna().unique() if isinstance(fac, str) and fac.strip() and fac not in HOSPITAL_LOCATIONS]
    if not unknown_facilities:
        return df

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception:
        return df

    prompt = f"""
    You are a geolocation expert for Ireland. Given a JSON list of Irish hospital abbreviations, your task is to identify their full official names and geographic coordinates.
    Your response MUST be a single, valid JSON object. Each key should be the original hospital abbreviation. The value should be another JSON object with keys "name", "lat", and "lon". If unknown, use null.
    **JSON List of Hospitals to Geocode:**
    {json.dumps(unknown_facilities)}
    **Your JSON Response:**
    """
    with st.spinner(f"AI is geolocating {len(unknown_facilities)} unknown facilities..."):
        try:
            response = model.generate_content(prompt)
            cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not cleaned_response: raise ValueError("AI geolocation response was not valid JSON.")
            location_data = json.loads(cleaned_response.group(0))
            for fac, data in location_data.items():
                if data and data.get('name'):
                    df.loc[df['HSE Facility'] == fac, ['name', 'lat', 'lon']] = [data['name'], data.get('lat'), data.get('lon')]
        except Exception as e:
            st.error(f"AI geolocation failed for unknown facilities: {e}")
    return df

def create_wc_image(text_series):
    full_text = ' '.join(text_series.dropna().astype(str))
    if not full_text: return None
    wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma', stopwords=list(stopwords.words('english'))).generate(full_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

def map_topical_category(category):
    if pd.isna(category): return 'Other'
    category_lower = str(category).lower()
    if 'incoming water supply' in category_lower: return 'Supply Infrastructure & Quality'
    if any(k in category_lower for k in ['distribution system', 'internal water', 'backup storage']): return 'Distribution & Internal Systems'
    if any(k in category_lower for k in ['metering', 'monitoring']): return 'Metering & Monitoring'
    if any(k in category_lower for k in ['protocol', 'eaps/sops', 'governance', 'communication']): return 'Protocols & Governance'
    if any(k in category_lower for k in ['maintenance', 'resources', 'procurement', 'contractor']): return 'Maintenance & Resources'
    if any(k in category_lower for k in ['documentation', 'drawings', 'maps']): return 'Data & Documentation'
    if any(k in category_lower for k in ['wastewater', 'stormwater']): return 'Wastewater & Stormwater'
    if any(k in category_lower for k in ['waste of water']): return 'Resource Management'
    return 'Other'

def run_dashboard():
    st.markdown(f'<div class="main-header"><h1>🏥 {AppConfig.APP_NAME}</h1><p>Version {AppConfig.APP_VERSION}</p></div>', unsafe_allow_html=True)
    download_nltk_stopwords()
    
    uploaded_file = st.file_uploader("Upload your HSE Risk Analysis Excel file", type="xlsx")
    if uploaded_file is None:
        st.info("Please upload a file to begin analysis.")
        return

    df = load_and_merge_data(uploaded_file)
    if df is None: return

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Gemini API Key is not configured. Please contact the administrator.")
        st.stop()

    df = assign_gemini_topics_batch(df, api_key)
    df = get_hospital_locations_batch(df, api_key)

    st.sidebar.header("Filter Options")
    
    ai_topic_options = sorted(df['AI-Generated Topic'].dropna().unique())
    selected_ai_topics = st.sidebar.multiselect("AI-Generated Topic:", options=ai_topic_options, default=ai_topic_options)
    
    source_tabs = st.sidebar.multiselect("HSE Facility:", options=df['HSE Facility'].unique(), default=df['HSE Facility'].unique())
    risk_locations = st.sidebar.multiselect("Risk Source Location:", options=df['Location of Risk Source'].dropna().unique(), default=df['Location of Risk Source'].dropna().unique())
    risk_ratings = st.sidebar.multiselect("Risk Rating:", options=df['Risk Rating'].dropna().unique(), default=df['Risk Rating'].dropna().unique())
    impact_categories = st.sidebar.multiselect("Impact Category:", options=df['Risk Impact Category'].dropna().unique(), default=df['Risk Impact Category'].dropna().unique())
    
    df_filtered = df[
        df['HSE Facility'].isin(source_tabs) &
        df['Risk Rating'].isin(risk_ratings) &
        df['Risk Impact Category'].isin(impact_categories) &
        df['Location of Risk Source'].isin(risk_locations) &
        df['AI-Generated Topic'].isin(selected_ai_topics)
    ].copy()
    
    df_filtered['Parent Category'] = df_filtered['Topical Category'].apply(map_topical_category)

    if df_filtered.empty:
        st.warning("No data matches the current filter settings.")
        return

    st.markdown('<div class="section-header"><h3>Filtered Risk Overview</h3></div>', unsafe_allow_html=True)
    st.metric(label="Total Risks Identified (Filtered)", value=f"{len(df_filtered)}")
    st.subheader("Risk Rating Breakdown")
    rating_counts = df_filtered['Risk Rating'].value_counts()
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="High Risks", value=rating_counts.get('High', 0))
    kpi2.metric(label="Medium Risks", value=rating_counts.get('Medium', 0))
    kpi3.metric(label="Low Risks", value=rating_counts.get('Low', 0))

    st.markdown('<div class="section-header"><h3>Distribution Analysis</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig_ai_donut = px.pie(df_filtered, names='AI-Generated Topic', hole=0.4, title="AI-Generated Topic Distribution")
        st.plotly_chart(fig_ai_donut, use_container_width=True)
    with col2:
        fig_rating = px.bar(df_filtered, x='Risk Rating', title="Risk Rating Distribution", color='Risk Rating')
        st.plotly_chart(fig_rating, use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        fig_impact = px.pie(df_filtered, names='Risk Impact Category', hole=0.4, title="Risk Impact Category Distribution")
        st.plotly_chart(fig_impact, use_container_width=True)
    with col4:
        location_counts = df_filtered['Location of Risk Source'].value_counts().reset_index()
        fig_location = px.treemap(location_counts, path=['Location of Risk Source'], values='count', title='Internal vs. External Risks')
        st.plotly_chart(fig_location, use_container_width=True)
    
    st.markdown('<div class="section-header"><h3>Hierarchical & Flow Analysis</h3></div>', unsafe_allow_html=True)
    col_hier1, col_hier2 = st.columns(2)
    with col_hier1:
        sunburst_data = df_filtered.dropna(subset=['Location of Risk Source', 'Risk Rating', 'Parent Category', 'Topical Category']).copy()
        if not sunburst_data.empty:
            for col in ['Parent Category', 'Topical Category']:
                 sunburst_data[col] = sunburst_data[col].astype(str).apply(lambda x: '<br>'.join(textwrap.wrap(x, width=20)))
            fig_sunburst = px.sunburst(sunburst_data, path=['Location of Risk Source', 'Risk Rating', 'Parent Category', 'Topical Category'], title="Sunburst View")
            st.plotly_chart(fig_sunburst, use_container_width=True)
    with col_hier2:
        sankey_data = df_filtered.dropna(subset=['Location of Risk Source', 'Risk Rating', 'Parent Category'])
        if not sankey_data.empty:
            all_nodes = list(pd.unique(sankey_data[['Location of Risk Source', 'Risk Rating', 'Parent Category']].values.ravel('K')))
            node_map = {node: i for i, node in enumerate(all_nodes)}
            palette = px.colors.qualitative.Plotly
            color_map = {node: palette[i % len(palette)] for i, node in enumerate(all_nodes)}
            
            links_1 = sankey_data.groupby(['Location of Risk Source', 'Risk Rating']).size().reset_index(name='value')
            links_2 = sankey_data.groupby(['Risk Rating', 'Parent Category']).size().reset_index(name='value')
            links = pd.concat([links_1, links_2], axis=0)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=25, thickness=20, line=dict(color="black", width=0.5), label=all_nodes, color=[color_map[n] for n in all_nodes]),
                link=dict(
                    source=links.iloc[:,0].map(node_map),
                    target=links.iloc[:,1].map(node_map),
                    value=links['value'],
                    color=[f"rgba({int(color_map[src][1:3], 16)}, {int(color_map[src][3:5], 16)}, {int(color_map[src][5:7], 16)}, 0.4)" for src in links.iloc[:,0]]
                )
            )])
            fig.update_layout(title_text="Risk Flow Analysis", font=dict(size=10), height=500)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header"><h3>Textual Insights</h3></div>', unsafe_allow_html=True)
    col_wc1, spacer, col_wc2 = st.columns([5,1,5])
    with col_wc1:
        st.subheader("Risk Description Word Cloud")
        if 'Risk Description' in df_filtered.columns:
            img_buf = create_wc_image(df_filtered['Risk Description'])
            if img_buf: st.image(img_buf)
    with col_wc2:
        st.subheader("Impact Description Word Cloud")
        if 'Impact Description' in df_filtered.columns:
            img_buf = create_wc_image(df_filtered['Impact Description'])
            if img_buf: st.image(img_buf)
    
    st.markdown('<div class="section-header"><h3>Geographic Risk Analysis</h3></div>', unsafe_allow_html=True)
    map_df = df_filtered.dropna(subset=['lat', 'lon'])
    if not map_df.empty:
        m = folium.Map(location=[53.4, -7.9], zoom_start=7)
        HeatMap(data=map_df[['lat', 'lon']].values.tolist(), radius=15).add_to(m)
        map_html = m._repr_html_()
        components.html(map_html, height=500)
    else:
        st.info("No geolocated data available for heatmap.")
    
    st.markdown("---")
    st.header("Filtered Data Details")
    st.dataframe(df_filtered)

def main():
    """Main function to handle authentication and run the app."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    def check_credentials():
        try:
            if st.session_state["password"] == st.secrets["PASSWORD"]:
                st.session_state.authenticated = True
            else:
                st.session_state.authenticated = False
        except KeyError:
            st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🔐 Login")
        st.text_input("Password", type="password", key="password")
        if st.button("Log in"):
            check_credentials()
            if not st.session_state.authenticated:
                st.error("😕 Invalid password")
            else:
                st.rerun()
    
    if st.session_state.authenticated:
        st.sidebar.success("Logged in successfully!")
        run_dashboard()

if __name__ == '__main__':
    main()

