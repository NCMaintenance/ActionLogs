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
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Risk Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Asset Downloading (Simplified) ---
@st.cache_resource
def download_nltk_stopwords():
    """Downloads NLTK stopwords if not already present."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        st.info("Downloading NLTK data (stopwords)...")
        nltk.download('stopwords')

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


# --- Helper Functions ---
def load_and_merge_data(uploaded_file):
    """
    Loads all tabs from an uploaded Excel file, merges them into a single
    DataFrame, and performs initial cleaning and transformations.
    """
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
                raise type(e)(f"{e} (while processing sheet: '{sheet_name}')")

        if all_sheets_df:
            merged_df = pd.concat(all_sheets_df, ignore_index=True)
            merged_df.columns = merged_df.columns.str.strip()

            if 'Risk Rating' in merged_df.columns:
                merged_df['Risk Rating'] = merged_df['Risk Rating'].str.strip()
                rating_map = {'Hign': 'High', 'Med': 'Medium'}
                merged_df['Risk Rating'].replace(rating_map, inplace=True)

            if 'Risk Impact Category' in merged_df.columns:
                merged_df['Risk Impact Category'] = merged_df['Risk Impact Category'].str.strip()
                merged_df['Risk Impact Category'].replace(to_replace=r'Loos of Trust / confidence|loss of Confidence', value='Loss of Confidence / Trust', regex=True, inplace=True)
                merged_df['Risk Impact Category'].replace(to_replace=r'Harm to Perso.*', value='Harm to Person', regex=True, inplace=True)

            if 'Location of Risk Source' in merged_df.columns:
                merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].astype(str).str.strip()
                merged_df = merged_df.assign(**{'Location of Risk Source': merged_df['Location of Risk Source'].str.split('/')}).explode('Location of Risk Source')
                merged_df['Location of Risk Source'] = merged_df['Location of Risk Source'].str.strip()
                # Clean up any blank entries that might result from splitting
                merged_df = merged_df[merged_df['Location of Risk Source'].str.strip() != '']
                merged_df.reset_index(drop=True, inplace=True)


            return merged_df
        else:
            st.warning("The uploaded Excel file seems to contain no data.")
            return None
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return None

# --- Gemini AI Integration ---
MAIN_CATEGORIES = [
    "Infrastructure, Equipment & Maintenance",
    "Water Quality & Pressure",
    "Governance, Communication & Procedures",
    "Procurement & Contractor Management",
    "Other"
]

@st.cache_data
def assign_gemini_topics_batch(_df, api_key):
    """
    Classifies a batch of unique 'Topical Category' values using a single
    Gemini API call for maximum efficiency.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception as e:
        st.error(f"🚨 API Key Configuration Error: {e}. Please ensure your key is valid.")
        st.stop()
    
    df = _df.copy()
    if 'Topical Category' not in df.columns:
        return df

    unique_topics = [topic for topic in df['Topical Category'].dropna().unique() if isinstance(topic, str) and topic.strip()]
    if not unique_topics:
        df['AI-Generated Topic'] = "Other"
        return df

    categories_str = "\n".join(f"- {cat}" for cat in MAIN_CATEGORIES)
    unique_topics_json = json.dumps(unique_topics)

    prompt = f"""
    You are an expert data classification system. Your task is to classify a list of topical issues into a set of predefined categories.
    Your response MUST be a single, valid JSON object where each key is the original issue text from the input list, and the corresponding value is ONLY the name of the most appropriate category from the predefined list. Do not add any other text, explanations, or formatting outside of the JSON object.
    **Predefined Categories:**
    {categories_str}
    **JSON List of Issues to Classify:**
    {unique_topics_json}
    **Your JSON Response:**
    """
    
    st.info(f"Sending {len(unique_topics)} unique categories to the AI for classification...")
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not cleaned_response:
            raise ValueError("The AI response did not contain a valid JSON object.")
        category_map = json.loads(cleaned_response.group(0))
    except Exception as e:
        st.error(f"An error occurred during AI topic classification: {e}")
        df['AI-Generated Topic'] = "Other"
        return df
    
    df['AI-Generated Topic'] = df['Topical Category'].map(category_map).fillna("Other")
    st.success(f"AI topic classification complete!")
    return df

@st.cache_data
def get_hospital_locations_batch(_df, api_key):
    """
    Gets full names and coordinates for HSE facilities using a combination of
    a manual lookup and a Gemini API call for unknown facilities.
    """
    df = _df.copy()
    
    df['name'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('name'))
    df['lat'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lat'))
    df['lon'] = df['HSE Facility'].map(lambda x: HOSPITAL_LOCATIONS.get(x, {}).get('lon'))
    
    unknown_facilities = [
        fac for fac in df['HSE Facility'].dropna().unique() 
        if isinstance(fac, str) and fac.strip() and fac not in HOSPITAL_LOCATIONS
    ]
    
    if not unknown_facilities:
        st.success("All hospital locations found in the predefined directory.")
        return df

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')
    except Exception:
        return df

    unique_facilities_json = json.dumps(unknown_facilities)
    prompt = f"""
    You are a geolocation expert for Ireland. Given a JSON list of Irish hospital abbreviations, your task is to identify their full official names and geographic coordinates.
    Your response MUST be a single, valid JSON object. Each key should be the original hospital abbreviation. The value should be another JSON object containing three keys: "name" (the full official hospital name), "lat" (latitude), and "lon" (longitude).
    If you cannot identify a hospital, use null for its values.

    **Example Input:**
    ["UHK", "SJH"]

    **Example Response:**
    {{
      "UHK": {{"name": "University Hospital Kerry", "lat": 52.268, "lon": -9.692}},
      "SJH": {{"name": "St. James's Hospital", "lat": 53.337, "lon": -6.301}}
    }}

    **JSON List of Hospitals to Geocode:**
    {unique_facilities_json}

    **Your JSON Response:**
    """

    st.info(f"Looking up {len(unknown_facilities)} unknown facilities with the AI...")

    try:
        response = model.generate_content(prompt)
        cleaned_response = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not cleaned_response:
            raise ValueError("The AI response for geolocation did not contain a valid JSON object.")
        
        location_data = json.loads(cleaned_response.group(0))
        
        for fac, data in location_data.items():
            if data and data.get('name'):
                df.loc[df['HSE Facility'] == fac, 'name'] = data['name']
                df.loc[df['HSE Facility'] == fac, 'lat'] = data['lat']
                df.loc[df['HSE Facility'] == fac, 'lon'] = data['lon']

    except Exception as e:
        st.error(f"An error occurred during AI geolocation for unknown facilities: {e}")
    
    st.success("AI geolocation complete!")
    return df


# --- Other Helper Functions ---
def create_wc_image(text_series):
    """Generates a classic word cloud image from a pandas Series of text."""
    full_text = ' '.join(text_series.dropna().astype(str))
    if not full_text:
        return None
    
    stop_words_list = list(stopwords.words('english'))
    wc = WordCloud(width=800, height=400, background_color='white', colormap='plasma', stopwords=stop_words_list).generate(full_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

def create_pie_chart_image(data, title):
    """Generates a static pie chart image as a base64 string."""
    if data.empty:
        return None
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(3.5, 3.5)) # Small figure size for popups
    ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
    ax.axis('equal')
    ax.set_title(title, fontsize=12)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2, transparent=True)
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode()

def map_topical_category(category):
    """Groups detailed topical categories into more granular and meaningful parent themes."""
    if pd.isna(category): return 'Other'
    category_lower = str(category).lower()
    if 'incoming water supply' in category_lower: return 'Supply Infrastructure & Quality'
    if any(k in category_lower for k in ['distribution system', 'internal water', 'backup storage', 'inability to promptly locate']): return 'Distribution & Internal Systems'
    if any(k in category_lower for k in ['metering', 'monitoring']): return 'Metering & Monitoring'
    if any(k in category_lower for k in ['protocol', 'eaps/sops', 'governance', 'committee', 'communication', 'lack of notice', 'delays in warning']): return 'Protocols & Governance'
    if any(k in category_lower for k in ['maintenance', 'resources', 'procurement', 'contracted pump']): return 'Maintenance & Resources'
    if any(k in category_lower for k in ['data and documentation', 'drawings, maps']): return 'Data & Documentation'
    if any(k in category_lower for k in ['wastewater', 'stormwater']): return 'Wastewater & Stormwater'
    if any(k in category_lower for k in ['waste of water', 'non-potable water capture']): return 'Resource Management'
    return 'Other'

def run_dashboard():
    """The main function to render the dashboard once authenticated."""
    st.title("📊 Interactive Risk Analysis Dashboard")
    st.markdown("Upload your Excel file to begin the analysis. The dashboard will update automatically.")
    download_nltk_stopwords()
    
    uploaded_file = st.file_uploader("Choose your 'Action Logs' Excel file", type="xlsx")
    if uploaded_file is None:
        st.info("Please upload an Excel file to proceed.")
        return

    df = load_and_merge_data(uploaded_file)
    if df is None: return

    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        st.error("Gemini API Key is not configured in the application secrets. Please contact the administrator.")
        st.stop()

    df = assign_gemini_topics_batch(df, api_key)
    df = get_hospital_locations_batch(df, api_key)

    st.sidebar.header("Filter Options")
    
    ai_topic_options = sorted(df['AI-Generated Topic'].dropna().unique())
    selected_ai_topics = st.sidebar.multiselect("Filter by AI-Generated Topic:", options=ai_topic_options, default=ai_topic_options)
    
    source_tabs = st.sidebar.multiselect("Filter by HSE Facility:", options=df['HSE Facility'].unique(), default=df['HSE Facility'].unique())
    risk_locations = st.sidebar.multiselect("Filter by Risk Source Location:", options=df['Location of Risk Source'].dropna().unique(), default=df['Location of Risk Source'].dropna().unique())
    risk_ratings = st.sidebar.multiselect("Filter by Risk Rating:", options=df['Risk Rating'].dropna().unique(), default=df['Risk Rating'].dropna().unique())
    impact_categories = st.sidebar.multiselect("Filter by Impact Category:", options=df['Risk Impact Category'].dropna().unique(), default=df['Risk Impact Category'].dropna().unique())
    
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

    st.header("Filtered Risk Overview")
    st.metric(label="Total Risks Identified (Filtered)", value=f"{len(df_filtered)}")
    st.markdown("---")
    
    st.subheader("Risk Rating Breakdown")
    rating_counts = df_filtered['Risk Rating'].value_counts()
    rating_percentages = (df_filtered['Risk Rating'].value_counts(normalize=True) * 100).round(1)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="🔴 High Risks", value=f"{rating_percentages.get('High', 0)}%", delta=f"{rating_counts.get('High', 0)} risks", delta_color="inverse")
    kpi2.metric(label="🟡 Medium Risks", value=f"{rating_percentages.get('Medium', 0)}%", delta=f"{rating_counts.get('Medium', 0)} risks", delta_color="inverse")
    kpi3.metric(label="🟢 Low Risks", value=f"{rating_percentages.get('Low', 0)}%", delta=f"{rating_counts.get('Low', 0)} risks", delta_color="off")
    st.markdown("---")

    st.header("Distribution Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("AI-Generated Topic Distribution")
        ai_topic_counts = df_filtered['AI-Generated Topic'].value_counts()
        fig_ai_donut = px.pie(
            values=ai_topic_counts.values,
            names=ai_topic_counts.index,
            hole=0.4,
            title="Distribution of AI-Generated Topics"
        )
        st.plotly_chart(fig_ai_donut, use_container_width=True)
    with col2:
        st.subheader("Risk Rating")
        rating_fig = px.bar(df_filtered['Risk Rating'].value_counts(), labels={'x': 'Risk Rating', 'y': 'Count'}, color=df_filtered['Risk Rating'].value_counts().index)
        rating_fig.update_layout(showlegend=False)
        st.plotly_chart(rating_fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Risk Impact Category")
        impact_fig = px.pie(df_filtered, names='Risk Impact Category', hole=0.4)
        st.plotly_chart(impact_fig, use_container_width=True)
    with col4:
        st.subheader("Location of Risk Source")
        location_counts = df_filtered['Location of Risk Source'].value_counts().reset_index()
        location_counts.columns = ['Location', 'Count']
        location_fig = px.treemap(location_counts, path=['Location'], values='Count', title='Internal (I) vs. External (E)')
        st.plotly_chart(location_fig, use_container_width=True)

    st.markdown("---")
    
    st.header("Hierarchical Risk Analysis (Manual Grouping)")
    
    col_hier1, col_hier2 = st.columns(2)
    with col_hier1:
        st.subheader("Sunburst View")
        sunburst_data = df_filtered.dropna(subset=['Location of Risk Source', 'Risk Rating', 'Parent Category', 'Topical Category'])
        if not sunburst_data.empty:
            fig_sunburst = px.sunburst(
                sunburst_data,
                path=['Location of Risk Source', 'Risk Rating', 'Parent Category', 'Topical Category'],
                title="Interactive Risk Breakdown"
            )
            fig_sunburst.update_layout(margin=dict(t=50, l=25, r=25, b=25))
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.info("Not enough data for Sunburst view.")
    with col_hier2:
        st.subheader("Risks by Facility")
        facility_counts = df_filtered['HSE Facility'].value_counts().reset_index()
        facility_counts.columns = ['HSE Facility', 'Number of Risks']
        fig_facility_treemap = px.treemap(
            facility_counts,
            path=['HSE Facility'],
            values='Number of Risks',
            title="Filtered Risks by HSE Facility"
        )
        fig_facility_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        st.plotly_chart(fig_facility_treemap, use_container_width=True)

    st.subheader("Risk Flow Analysis")
    sankey_data = df_filtered.dropna(subset=['Location of Risk Source', 'Risk Rating', 'Parent Category'])
    if not sankey_data.empty:
        # Robustly create the list of nodes, ensuring they are all strings
        nodes_l0 = sorted(sankey_data['Location of Risk Source'].astype(str).unique())
        nodes_l1 = sorted(sankey_data['Risk Rating'].astype(str).unique())
        nodes_l2 = sorted(sankey_data['Parent Category'].astype(str).unique())
        
        all_nodes = nodes_l0 + nodes_l1 + nodes_l2
        node_map = {node: i for i, node in enumerate(all_nodes)}
        
        palette = px.colors.qualitative.Plotly
        color_map = {node: palette[i % len(palette)] for i, node in enumerate(all_nodes)}
        node_colors = [color_map[node] for node in all_nodes]

        sankey_df_1 = sankey_data.groupby(['Location of Risk Source', 'Risk Rating']).size().reset_index(name='value')
        sankey_df_1['source'] = sankey_df_1['Location of Risk Source'].map(node_map)
        sankey_df_1['target'] = sankey_df_1['Risk Rating'].map(node_map)
        
        sankey_df_2 = sankey_data.groupby(['Risk Rating', 'Parent Category']).size().reset_index(name='value')
        sankey_df_2['source'] = sankey_df_2['Risk Rating'].map(node_map)
        sankey_df_2['target'] = sankey_df_2['Parent Category'].map(node_map)
        
        links = pd.concat([sankey_df_1, sankey_df_2], axis=0).dropna(subset=['source', 'target'])
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=25,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color=node_colors,
                hovertemplate='%{label} has %{value} risks<extra></extra>'
            ),
            link=dict(
                source=links['source'].astype(int),
                target=links['target'].astype(int),
                value=links['value']
            ))])
        fig.update_layout(title_text="Risk Flow: Source -> Rating -> Category", font_size=10, height=600, margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.header("Textual Insights via Word Clouds")
    col_wc1, col_wc2 = st.columns(2)
    with col_wc1:
        st.subheader("Risk Description")
        if 'Risk Description' in df_filtered.columns:
            img_buf = create_wc_image(df_filtered['Risk Description'])
            if img_buf:
                st.image(img_buf)
            else:
                st.info("Not enough data for Risk Description word cloud.")
    with col_wc2:
        st.subheader("Impact Description")
        if 'Impact Description' in df_filtered.columns:
            img_buf = create_wc_image(df_filtered['Impact Description'])
            if img_buf:
                st.image(img_buf)
            else:
                st.info("Not enough data for Impact Description word cloud.")
    st.markdown("---")

    st.header("Geographic Risk Analysis")
    map_df = df_filtered.dropna(subset=['lat', 'lon'])
    
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.subheader("Risk Heatmap")
        if not map_df.empty:
            m1 = folium.Map(location=[53.4, -7.9], zoom_start=7)
            HeatMap(data=map_df[['lat', 'lon']].values.tolist(), radius=15).add_to(m1)
            folium_static(m1, key="heatmap")
        else:
            st.info("No geolocated data for heatmap.")
    
    with col_map2:
        st.subheader("AI Topics by Hospital")
        if not map_df.empty:
            m2 = folium.Map(location=[53.4, -7.9], zoom_start=7)
            locations = map_df.groupby(['HSE Facility', 'name', 'lat', 'lon']).size().reset_index(name='count')
            for idx, row in locations.iterrows():
                hospital_data = map_df[map_df['HSE Facility'] == row['HSE Facility']]
                ai_topic_counts = hospital_data['AI-Generated Topic'].value_counts()
                
                b64_image = create_pie_chart_image(ai_topic_counts, title="AI Topics")
                
                if b64_image:
                    html = f"""
                    <b>{row['name']}</b><br>
                    Total Risks: {row['count']}<br>
                    <img src="data:image/png;base64,{b64_image}">
                    """
                    popup = folium.Popup(html, max_width=400)
                    folium.Marker(
                        [row['lat'], row['lon']],
                        popup=popup,
                        tooltip=row['name']
                    ).add_to(m2)
            folium_static(m2, key="piechart_map")
        else:
            st.info("No geolocated data for pie chart map.")

    st.markdown("---")
    
    st.header("Filtered Data Details")
    st.dataframe(df_filtered)

def main():
    """Main function to handle authentication and run the app."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    def check_credentials():
        """Validates credentials against st.secrets."""
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

