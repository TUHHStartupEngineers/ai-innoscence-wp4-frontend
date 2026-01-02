"""
Circular Economy Ecosystem Visualization
Interactive Streamlit application for exploring CE ecosystems
Part of the AI-InnoScEnCE Project

Supports: Hamburg (Germany), Novi Sad (Serbia), Cahul (Moldova)

Deployed on Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import ast
import sqlite3
from datetime import datetime

# Import taxonomy from local config package
from config.ce_activities_taxonomy import CE_ACTIVITIES_TAXONOMY, get_all_activities, get_activity_categories

# =============================================================================
# ECOSYSTEM CONFIGURATION
# =============================================================================
ECOSYSTEMS = {
    "Hamburg": {
        "name": "Hamburg",
        "country": "Germany",
        "db_path": "data/hamburg.db",
        "center_lat": 53.5511,
        "center_lon": 9.9937,
        "zoom": 11,
        # Geographic bounds for filtering (approx 30km radius)
        "lat_min": 53.3, "lat_max": 53.8,
        "lon_min": 9.7, "lon_max": 10.3,
    },
    "Novi Sad": {
        "name": "Novi Sad",
        "country": "Serbia",
        "db_path": "data/novi_sad.db",
        "center_lat": 45.2671,
        "center_lon": 19.8335,
        "zoom": 12,
        # Geographic bounds for filtering (approx 30km radius)
        "lat_min": 45.1, "lat_max": 45.5,
        "lon_min": 19.5, "lon_max": 20.1,
    },
    "Cahul": {
        "name": "Cahul",
        "country": "Moldova",
        "db_path": "data/cahul.db",
        "center_lat": 45.9042,
        "center_lon": 28.1994,
        "zoom": 13,
        # Geographic bounds for filtering (approx 30km radius)
        "lat_min": 45.7, "lat_max": 46.1,
        "lon_min": 27.9, "lon_max": 28.5,
    },
}

# Page configuration
st.set_page_config(
    page_title="CE Ecosystem | AI-InnoScEnCE",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - AI-InnoScEnCE Branding (Light Theme)
st.markdown("""
<style>
    /* Main content area */
    .main {
        background-color: #FFFFFF;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        margin-bottom: 1rem;
    }

    /* Metric cards */
    .metric-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00BCD4;
    }
    .stMetric {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }

    /* Buttons - Turquoise theme */
    .stButton>button {
        background-color: #00BCD4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00ACC1;
        border: none;
        box-shadow: 0 4px 8px rgba(0,188,212,0.3);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F5F5F5;
        padding: 0.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #1e3a5f;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00BCD4;
        color: white !important;
        border-radius: 6px;
    }

    /* Divider */
    .divider {
        margin: 2rem 0;
        border-top: 2px solid #e0e0e0;
    }

    /* Insight cards */
    .insight-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #00BCD4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        word-wrap: break-word;
        overflow-wrap: break-word;
        white-space: normal;
        max-height: none;
        overflow: visible;
    }
    .gap-card {
        border-left-color: #ff6b6b;
        background-color: #fff5f5;
    }
    .synergy-card {
        border-left-color: #4ecdc4;
        background-color: #f0fffe;
    }
    .recommendation-card {
        border-left-color: #95e1d3;
        background-color: #f7fffe;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F5F5F5;
    }

    /* Dataframes and tables */
    .stDataFrame {
        background-color: #FFFFFF;
    }

    /* Headers in markdown */
    h1, h2, h3 {
        color: #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)

# Database connectivity
def get_db_connection(ecosystem: str):
    """Get SQLite database connection for the specified ecosystem"""
    app_dir = Path(__file__).parent
    ecosystem_config = ECOSYSTEMS.get(ecosystem, ECOSYSTEMS["Hamburg"])
    db_path = app_dir / ecosystem_config["db_path"]

    if not db_path.exists():
        st.error(f"Database not found at: {db_path}")
        return None

    return sqlite3.connect(str(db_path), check_same_thread=False)

@st.cache_data
def load_relationships(ecosystem: str):
    """Load relationships from database"""
    conn = get_db_connection(ecosystem)
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT source_entity, target_entity, relationship_type, confidence,
           evidence, bidirectional, source_url, target_url
    FROM relationships
    ORDER BY confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def load_clusters(ecosystem: str):
    """Load clusters from database"""
    conn = get_db_connection(ecosystem)
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT cluster_id, cluster_name, cluster_type, description,
           entities, items, confidence
    FROM clusters
    ORDER BY cluster_type, confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def load_insights(ecosystem: str):
    """Load ecosystem insights from database"""
    conn = get_db_connection(ecosystem)
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT insight_type, title, description, entities_involved,
           confidence, priority, timestamp
    FROM ecosystem_insights
    ORDER BY
        CASE priority
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END,
        confidence DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

@st.cache_data
def load_entities(ecosystem: str):
    """Load entities from database (deduplicated)"""
    conn = get_db_connection(ecosystem)
    if conn is None:
        return pd.DataFrame()

    query = """
    SELECT url, entity_name, ecosystem_role, brief_description,
           ce_relation, ce_activities, capabilities_offered,
           needs_requirements, capability_categories, partners, partner_urls,
           address, latitude, longitude, extraction_confidence,
           ce_capabilities_offered, ce_needs_requirements
    FROM entity_profiles
    """
    df = pd.read_sql_query(query, conn)

    # Clean up data
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Parse list-like strings
    def safe_parse_list(val):
        if pd.isna(val) or val in ['[]', '', 'NA']:
            return []
        try:
            return ast.literal_eval(val) if isinstance(val, str) else val
        except:
            return []

    # Parse JSON fields
    def safe_parse_json(val):
        if pd.isna(val) or val in ['[]', '', 'NA', None]:
            return []
        try:
            if isinstance(val, str):
                return json.loads(val)
            return val
        except:
            return []

    for col in ['ce_activities', 'partners']:
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(safe_parse_list)
            df[f'{col}_count'] = df[f'{col}_parsed'].apply(len)

    # Parse capabilities and needs (JSON format)
    if 'ce_capabilities_offered' in df.columns:
        df['capabilities_parsed'] = df['ce_capabilities_offered'].apply(safe_parse_json)
        df['capabilities_count'] = df['capabilities_parsed'].apply(len)

    if 'ce_needs_requirements' in df.columns:
        df['needs_parsed'] = df['ce_needs_requirements'].apply(safe_parse_json)
        df['needs_count'] = df['needs_parsed'].apply(len)

    # Clean entity names and roles
    df['entity_name'] = df['entity_name'].fillna('Unknown')
    df['ecosystem_role'] = df['ecosystem_role'].fillna('Unknown')

    # Filter out invalid roles (data artifacts)
    valid_roles = [
        'Industry Partners', 'Higher Education Institutions', 'Startups and Entrepreneurs',
        'Non-Governmental Organizations', 'Research Institutes', 'Researchers',
        'Media and Communication Partners', 'Citizen Associations', 'End-Users',
        'Public Authorities', 'Knowledge and Innovation Communities', 'Students',
        'Policy Makers', 'Funding Bodies'
    ]
    df = df[df['ecosystem_role'].isin(valid_roles + ['Unknown', ''])]

    conn.close()
    return df

# Define color mapping for ecosystem roles
ROLE_COLORS = {
    'Industry Partners': [0, 128, 255],  # Blue
    'Higher Education Institutions': [255, 140, 0],  # Dark Orange
    'Startups and Entrepreneurs': [233, 30, 99],  # Pink/Magenta
    'Non-Governmental Organizations': [155, 89, 182],  # Purple
    'Research Institutes': [231, 76, 60],  # Red
    'Researchers': [241, 196, 15],  # Yellow
    'Media and Communication Partners': [26, 188, 156],  # Turquoise
    'Citizen Associations': [230, 126, 34],  # Orange
    'End-Users': [149, 165, 166],  # Gray
    'Public Authorities': [52, 73, 94],  # Dark Blue
    'Knowledge and Innovation Communities': [142, 68, 173],  # Purple
    'Students': [255, 193, 7],  # Amber/Gold
    'Policy Makers': [192, 57, 43],  # Dark Red
    'Funding Bodies': [103, 58, 183],  # Deep Purple
    'Unknown': [200, 200, 200],  # Light Gray
}

def main():
    # Header with logo
    app_dir = Path(__file__).parent
    logo_path = app_dir / "assets" / "AI-INNOCENSE-LOGO.png"

    col1, col2 = st.columns([1, 6])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=180)
    with col2:
        st.markdown('<p class="main-header">Circular Economy Ecosystem Explorer</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Empowered Innovation in Natural Science and Engineering for the Circular Economy</p>', unsafe_allow_html=True)

    # Ecosystem selector
    selected_ecosystem = st.selectbox(
        "Select Ecosystem",
        options=list(ECOSYSTEMS.keys()),
        index=0,
        help="Switch between different CE ecosystems"
    )

    ecosystem_config = ECOSYSTEMS[selected_ecosystem]
    st.markdown(f"**Currently viewing:** {ecosystem_config['name']}, {ecosystem_config['country']}")

    st.markdown("---")

    # Load data from database (deduplicated)
    with st.spinner(f"Loading {selected_ecosystem} ecosystem data..."):
        df = load_entities(selected_ecosystem)
        if df.empty:
            st.error("Failed to load entities from database")
            return
        relationships_df = load_relationships(selected_ecosystem)
        clusters_df = load_clusters(selected_ecosystem)
        insights_df = load_insights(selected_ecosystem)

    # Apply geographic bounds filtering to remove entities outside ecosystem area
    # This ensures only entities actually in the ecosystem region are counted/displayed
    if 'lat_min' in ecosystem_config and not df.empty:
        has_coords = df['latitude'].notna() & df['longitude'].notna()
        in_bounds = (
            (df['latitude'] >= ecosystem_config['lat_min']) &
            (df['latitude'] <= ecosystem_config['lat_max']) &
            (df['longitude'] >= ecosystem_config['lon_min']) &
            (df['longitude'] <= ecosystem_config['lon_max'])
        )
        # Keep entities that either have no coords OR are within bounds
        df = df[~has_coords | in_bounds].copy()

    # Create entity-to-URL mapping for relationships links
    entity_url_map = df.set_index('entity_name')['url'].to_dict() if 'url' in df.columns else {}

    # Sidebar filters
    st.sidebar.header("Filters")

    # Role filter
    all_roles = sorted(df['ecosystem_role'].dropna().unique())
    selected_roles = st.sidebar.multiselect(
        "Ecosystem Roles",
        options=all_roles,
        default=all_roles  # Show ALL roles by default
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_roles:
        filtered_df = filtered_df[filtered_df['ecosystem_role'].isin(selected_roles)]

    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.header("Statistics")
    # Count entities that can appear on map (have valid coordinates)
    mappable_count = len(df[df['latitude'].notna() & df['longitude'].notna()])
    st.sidebar.metric("Total Entities", f"{len(df):,}")
    st.sidebar.metric("On Map", f"{mappable_count:,}")
    partnerships_count = len(relationships_df[relationships_df['relationship_type'] == 'partnership'])
    st.sidebar.metric("Partnerships", f"{partnerships_count:,}")
    st.sidebar.metric("Clusters", f"{len(clusters_df):,}")

    # Main navigation using radio (persists selection across reruns)
    selected_tab = st.radio(
        "Navigation",
        options=["Map View", "Entities", "Partnerships", "Insights", "Collaboration", "About"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_nav"
    )

    st.markdown("---")

    # Display content based on selected tab
    if selected_tab == "Map View":
        show_map_view(filtered_df, clusters_df, ecosystem_config)
    elif selected_tab == "Entities":
        show_entities_browser(df)
    elif selected_tab == "Partnerships":
        show_relationships_tab(df, relationships_df, entity_url_map)
    elif selected_tab == "Insights":
        show_insights_dashboard(insights_df)
    elif selected_tab == "Collaboration":
        show_collaboration_finder(df, relationships_df, clusters_df)
    elif selected_tab == "About":
        show_about(df, relationships_df, clusters_df, insights_df)

    # EU Funding acknowledgment footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.85rem;">'
        'This project has received funding from the European Union\'s Horizon Europe '
        'research and innovation programme and is part of the EIT HEI Initiative. '
        '</p>',
        unsafe_allow_html=True
    )

def show_entities_browser(df):
    """Browse all ecosystem entities by role"""
    st.header("Ecosystem Entities")

    st.markdown("""
    Browse all entities in this ecosystem, including those without map coordinates.
    Filter by role or search by name.
    """)

    # Filters in columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Role filter
        all_roles = sorted(df['ecosystem_role'].dropna().unique())
        selected_role = st.selectbox(
            "Filter by Role",
            options=["All Roles"] + all_roles,
            key="entity_role_filter"
        )

    with col2:
        # Search by name
        search_term = st.text_input(
            "Search by Name",
            placeholder="Type to search...",
            key="entity_search"
        )

    # Apply filters
    filtered = df.copy()
    if selected_role != "All Roles":
        filtered = filtered[filtered['ecosystem_role'] == selected_role]
    if search_term:
        filtered = filtered[filtered['entity_name'].str.contains(search_term, case=False, na=False)]

    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Matching Entities", len(filtered))
    with col2:
        with_coords = len(filtered[filtered['latitude'].notna()])
        st.metric("With Coordinates", with_coords)
    with col3:
        without_coords = len(filtered[filtered['latitude'].isna()])
        st.metric("Without Coordinates", without_coords)

    st.markdown("---")

    if len(filtered) == 0:
        st.info("No entities match your filter criteria.")
        return

    # Display entities as expandable cards
    for _, entity in filtered.head(100).iterrows():
        # Determine map status
        has_coords = pd.notna(entity.get('latitude')) and pd.notna(entity.get('longitude'))
        map_status = "On Map" if has_coords else "Not Mapped"

        with st.expander(f"**{entity['entity_name']}** | {entity['ecosystem_role']} | {map_status}"):
            col1, col2 = st.columns([2, 1])

            with col1:
                if pd.notna(entity.get('brief_description')):
                    st.markdown("**Description:**")
                    st.write(entity['brief_description'])

                # CE Activities
                activities = entity.get('ce_activities_parsed', [])
                if activities:
                    st.markdown(f"**CE Activities ({len(activities)}):**")
                    st.write(", ".join(activities[:5]))
                    if len(activities) > 5:
                        st.caption(f"... and {len(activities) - 5} more")

            with col2:
                st.markdown("**Status:**")
                if has_coords:
                    st.success(f"Mapped")
                    st.caption(f"Lat: {entity['latitude']:.4f}")
                    st.caption(f"Lon: {entity['longitude']:.4f}")
                else:
                    st.warning("No coordinates")
                    if pd.notna(entity.get('address')):
                        st.caption(f"Address: {entity['address'][:50]}...")

                # Capabilities count
                caps = entity.get('capabilities_parsed', [])
                needs = entity.get('needs_parsed', [])
                if caps:
                    st.caption(f"{len(caps)} capabilities")
                if needs:
                    st.caption(f"{len(needs)} needs")

    if len(filtered) > 100:
        st.info(f"Showing first 100 of {len(filtered)} entities.")


def show_map_view(df, clusters_df, ecosystem_config):
    """Display interactive map with cluster overlay"""
    st.header(f"Interactive {ecosystem_config['name']} CE Ecosystem Map")

    # Filter out rows without coordinates
    map_df = df.dropna(subset=['latitude', 'longitude']).copy()

    # Filter to only entities within the ecosystem's geographic bounds
    if 'lat_min' in ecosystem_config:
        map_df = map_df[
            (map_df['latitude'] >= ecosystem_config['lat_min']) &
            (map_df['latitude'] <= ecosystem_config['lat_max']) &
            (map_df['longitude'] >= ecosystem_config['lon_min']) &
            (map_df['longitude'] <= ecosystem_config['lon_max'])
        ]

    if len(map_df) == 0:
        st.warning("No entities with coordinates in the current filter selection.")
        return

    # Add color based on role
    map_df['color'] = map_df['ecosystem_role'].map(lambda r: ROLE_COLORS.get(r, [200, 200, 200]))

    # Create tooltip with full CE activities list
    def format_tooltip(row):
        activities_list = row.get('ce_activities_parsed', [])
        if activities_list and len(activities_list) > 0:
            activities_html = '<br/>  - ' + '<br/>  - '.join(activities_list)
            activities_section = f"CE Activities ({len(activities_list)}):{activities_html}"
        else:
            activities_section = "CE Activities: None"

        return f"""
        <b>{row['entity_name']}</b><br/>
        Role: {row['ecosystem_role']}<br/>
        Location: Lat {row['latitude']:.6f}, Lon {row['longitude']:.6f}<br/>
        {activities_section}<br/>
        Partners: {row.get('partners_count', 0)}
        """

    map_df['tooltip'] = map_df.apply(format_tooltip, axis=1)

    # Map configuration - use ecosystem-specific center coordinates
    view_state = pdk.ViewState(
        latitude=ecosystem_config['center_lat'],
        longitude=ecosystem_config['center_lon'],
        zoom=ecosystem_config['zoom'],
        pitch=0,  # 0 = flat 2D view, 45 = 3D tilted view
    )

    # Layer configuration
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=150,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=50,
        line_width_min_pixels=1,
    )

    # Render map
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"html": "<b>{tooltip}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}},
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",  # Free Carto basemap (no token required)
    )

    st.pydeck_chart(r)

    # Legend
    st.markdown("### Legend")
    cols = st.columns(5)
    for idx, (role, color) in enumerate(ROLE_COLORS.items()):
        if role in df['ecosystem_role'].values:
            count = len(df[df['ecosystem_role'] == role])
            cols[idx % 5].markdown(
                f'<div style="display: flex; align-items: center; margin: 0.2rem 0;">'
                f'<div style="width: 15px; height: 15px; min-width: 15px; min-height: 15px; '
                f'background-color: rgb({color[0]}, {color[1]}, {color[2]}); '
                f'border-radius: 50%; margin-right: 0.5rem; flex-shrink: 0; display: inline-block;"></div>'
                f'<span>{role} ({count})</span></div>',
                unsafe_allow_html=True
            )

def show_about(df, relationships_df, clusters_df, insights_df):
    """Display about information"""
    st.header("About This Application")

    st.markdown("""
    ### Circular Economy Ecosystem Explorer

    This interactive application visualizes Circular Economy (CE) ecosystems across Europe,
    built using **ScrapegraphAI** and advanced LLM-based extraction techniques as part of the
    **AI-InnoScEnCE Project**.

    **Supported Ecosystems:** Hamburg (Germany), Novi Sad (Serbia), Cahul (Moldova)

    #### Data Overview""")

    # Dynamic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entities", f"{len(df):,}")
        partnerships = len(relationships_df[relationships_df['relationship_type'] == 'partnership'])
        st.metric("Verified Partnerships", f"{partnerships:,}")
    with col2:
        st.metric("Clusters", f"{len(clusters_df):,}")
        st.metric("Insights", f"{len(insights_df):,}")
    with col3:
        st.metric("CE Activity Categories", "10")
        st.metric("Predefined Activities", "120")

    st.markdown("""
    - Data extracted from: websites, Impressum pages, contact forms, and more
    - Privacy-compliant: No sensitive contact information is displayed

    #### Ecosystem Roles

    The ecosystem comprises various stakeholders:
    - **Industry Partners** - Companies and manufacturers
    - **Higher Education Institutions** - Universities and colleges
    - **Startups and Entrepreneurs** - Innovative ventures and new businesses
    - **Non-Governmental Organizations** - NGOs and civil society organizations
    - **Research Institutes** - Research centers and laboratories
    - **Researchers** - Individual researchers and scientists
    - **Media and Communication Partners** - Press, media, and communication organizations
    - **Citizen Associations** - Community groups and local associations
    - **End-Users** - Consumers and end-users of circular products
    - **Public Authorities** - Government bodies and public agencies
    - **Knowledge and Innovation Communities** - KICs and innovation hubs
    - **Students** - Students and academic learners
    - **Policy Makers** - Policy developers and decision makers
    - **Funding Bodies** - Investors, grants, and funding organizations

    #### Technology Stack

    - **ScrapegraphAI** - AI-powered web scraping
    - **LLM Extraction** - Ollama-based structured extraction
    - **Geocoding** - Nominatim with intelligent fallback strategies
    - **Visualization** - Streamlit + PyDeck + Plotly

    #### Data Pipeline

    1. **Verification** - LLM checks location & CE relevance
    2. **Extraction** - ScrapegraphAI + Ollama extract entity data
    3. **Geocoding** - Multi-strategy geocoding with caching
    4. **Visualization** - This Streamlit application

    #### Contact & Support

    For questions, suggestions, or collaboration opportunities, please visit:
    **[https://ai-innoscence.eu/](https://ai-innoscence.eu/)**

    ---

    **Built for a more circular Europe**
    """)

def show_relationships_tab(df, relationships_df, entity_url_map):
    """Display verified partnerships within the CE ecosystem"""
    st.header("Verified Partnerships")

    st.markdown("""
    Browse verified partnerships between entities in the CE ecosystem.
    These are real collaborations identified from entity websites and data sources.
    """)

    # Show partnerships directly (no nested tabs)
    show_partnerships_view(relationships_df, entity_url_map)


def show_partnerships_view(relationships_df, entity_url_map):
    """Display partnerships table"""
    st.subheader("Verified Partnerships")

    # Filter for partnerships only and exclude self-referencing relationships
    partnerships = relationships_df[relationships_df['relationship_type'] == 'partnership'].copy()
    partnerships = partnerships[partnerships['source_entity'] != partnerships['target_entity']]

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Partnerships", len(partnerships))
    with col2:
        avg_conf = partnerships['confidence'].mean() if len(partnerships) > 0 else 0
        st.metric("Average Confidence", f"{avg_conf:.0%}")
    with col3:
        unique_entities = len(set(partnerships['source_entity'].tolist() + partnerships['target_entity'].tolist()))
        st.metric("Entities Involved", unique_entities)

    st.markdown("---")

    if len(partnerships) == 0:
        st.info("No partnerships found in the database.")
        return

    # Search functionality
    search = st.text_input("Search partnerships", placeholder="Search by entity name...")

    if search:
        mask = (
            partnerships['source_entity'].str.contains(search, case=False, na=False) |
            partnerships['target_entity'].str.contains(search, case=False, na=False) |
            partnerships['evidence'].str.contains(search, case=False, na=False)
        )
        partnerships = partnerships[mask]

    # Sort by confidence
    partnerships = partnerships.sort_values('confidence', ascending=False)

    # Display partnerships as expandable cards
    st.markdown(f"### {len(partnerships)} Partnership{'s' if len(partnerships) != 1 else ''}")

    for idx, row in partnerships.iterrows():
        with st.expander(f"**{row['source_entity']}** <-> **{row['target_entity']}** | Confidence: {row['confidence']:.0%}"):
            st.markdown(f"**Evidence:**")
            st.write(row['evidence'])  # Full evidence text, not truncated

            # Use URLs from relationships table, fallback to entity URL mapping
            source_url = row.get('source_url') if pd.notna(row.get('source_url')) else entity_url_map.get(row['source_entity'])
            target_url = row.get('target_url') if pd.notna(row.get('target_url')) else entity_url_map.get(row['target_entity'])

            if source_url:
                st.markdown(f"[Visit {row['source_entity']}]({source_url})")
            if target_url:
                st.markdown(f"[Visit {row['target_entity']}]({target_url})")


def show_insights_dashboard(insights_df):
    """Display ecosystem insights dashboard"""
    st.header("Ecosystem Insights")

    st.markdown("""
    AI-generated insights about the Hamburg CE ecosystem, including identified gaps,
    potential synergies, and recommendations for ecosystem development.
    """)

    if len(insights_df) == 0:
        st.warning("No insights available in the database.")
        return

    # Tabs for different insight types
    insight_tabs = st.tabs(["All Insights", "Gaps", "Synergies", "Recommendations"])

    with insight_tabs[0]:
        show_all_insights(insights_df)

    with insight_tabs[1]:
        show_insights_by_type(insights_df, 'gap', 'gap-card')

    with insight_tabs[2]:
        show_insights_by_type(insights_df, 'synergy', 'synergy-card')

    with insight_tabs[3]:
        show_insights_by_type(insights_df, 'recommendation', 'recommendation-card')

def show_all_insights(insights_df):
    """Display all insights"""
    st.subheader("All Ecosystem Insights")

    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        gaps = len(insights_df[insights_df['insight_type'] == 'gap'])
        st.metric("Gaps Identified", gaps)
    with col2:
        synergies = len(insights_df[insights_df['insight_type'] == 'synergy'])
        st.metric("Synergies Found", synergies)
    with col3:
        recommendations = len(insights_df[insights_df['insight_type'] == 'recommendation'])
        st.metric("Recommendations", recommendations)

    st.markdown("---")

    # Display all insights
    for _, insight in insights_df.iterrows():
        card_class = {
            'gap': 'gap-card',
            'synergy': 'synergy-card',
            'recommendation': 'recommendation-card'
        }.get(insight['insight_type'], 'insight-card')

        priority_badge = f"HIGH" if insight['priority'] == 'high' else "MEDIUM"
        confidence_badge = f"Confidence: {insight['confidence']:.0%}"

        st.markdown(f"""
        <div class="insight-card {card_class}">
            <h4>{insight['title']}</h4>
            <p><strong>{priority_badge}</strong> | {confidence_badge} | Type: {insight['insight_type'].title()}</p>
            <p>{insight['description']}</p>
        </div>
        """, unsafe_allow_html=True)

def show_insights_by_type(insights_df, insight_type, card_class):
    """Display insights filtered by type"""
    filtered = insights_df[insights_df['insight_type'] == insight_type]

    if len(filtered) == 0:
        st.info(f"No {insight_type}s found in the database.")
        return

    st.subheader(f"{len(filtered)} {insight_type.title()}{'s' if len(filtered) != 1 else ''} Identified")

    for _, insight in filtered.iterrows():
        priority_badge = f"HIGH PRIORITY" if insight['priority'] == 'high' else "MEDIUM PRIORITY"
        confidence_badge = f"Confidence: {insight['confidence']:.0%}"

        # Parse entities if it's a JSON string
        entities_text = ""
        try:
            entities = json.loads(insight['entities_involved']) if isinstance(insight['entities_involved'], str) else insight['entities_involved']
            if entities and len(entities) > 0:
                # Show ALL entities, not truncated
                entities_text = f"<p><strong>Entities Involved:</strong> {', '.join(entities)}</p>"
        except:
            pass

        st.markdown(f"""
        <div class="insight-card {card_class}">
            <h4>{insight['title']}</h4>
            <p>{priority_badge} | {confidence_badge}</p>
            <p>{insight['description']}</p>
            {entities_text}
        </div>
        """, unsafe_allow_html=True)

def show_collaboration_finder(df, relationships_df, clusters_df):
    """Display collaboration finder tool"""
    st.header("Collaboration Finder")

    st.markdown("""
    Find potential collaboration opportunities by browsing CE activities, capabilities, needs,
    or exploring entity clusters.
    """)

    # Search tabs - separate tabs for Activities, Capabilities, Needs, and Clusters
    search_tabs = st.tabs(["Activities", "Capabilities", "Needs", "By Cluster"])

    with search_tabs[0]:
        show_taxonomy_browser(df)

    with search_tabs[1]:
        show_capabilities_browser(df)

    with search_tabs[2]:
        show_needs_browser(df)

    with search_tabs[3]:
        show_cluster_collaboration(clusters_df)

def show_taxonomy_browser(df):
    """Browse CE activities by taxonomy category and find matching entities"""
    st.subheader("Browse CE Activities Taxonomy")

    # Definition for CE Activities
    with st.expander("What is a CE Activity?", expanded=False):
        st.markdown("""
        A **CE Activity (Circular Economy Activity)** refers to any practice, action, or service
        that contributes to the circular economy. These activities aim to reduce waste, reuse
        products, recycle materials, and recover value from waste streams.

        *Examples: recycling services, repair workshops, waste collection, eco-design,
        industrial symbiosis, sharing platforms.*

        The taxonomy below organizes **120 standardized CE activities** into 10 categories
        based on the Ellen MacArthur Foundation framework and EU Circular Economy Action Plan.
        """)

    st.markdown("""
    Browse the standardized CE activities taxonomy (120 activities in 10 categories).
    Select a category and activity to find entities engaged in that area.
    """)

    # Get taxonomy data
    categories = get_activity_categories()
    all_taxonomy_activities = set(get_all_activities())

    # Category dropdown
    selected_category = st.selectbox(
        "Select Category",
        options=categories,
        help="Choose a CE activity category"
    )

    # Get activities for selected category
    category_activities = CE_ACTIVITIES_TAXONOMY.get(selected_category, [])

    # Activity dropdown
    selected_activity = st.selectbox(
        "Select Activity",
        options=["-- All activities in category --"] + category_activities,
        help="Choose a specific activity to find matching entities"
    )

    st.markdown("---")

    # Find entities with matching activities
    if selected_activity == "-- All activities in category --":
        # Match any activity in the category
        activities_to_match = set(category_activities)
    else:
        activities_to_match = {selected_activity}

    # Filter entities
    matching_entities = df[df['ce_activities_parsed'].apply(
        lambda activities: bool(set(activities) & activities_to_match) if activities else False
    )]

    # Display results
    st.metric("Matching Entities", len(matching_entities))

    if len(matching_entities) == 0:
        st.info("No entities found with the selected activity.")
        return

    # Display as expandable cards
    for _, entity in matching_entities.head(50).iterrows():
        # Get matching activities for this entity
        entity_activities = entity.get('ce_activities_parsed', [])
        matched = [a for a in entity_activities if a in activities_to_match]

        with st.expander(f"**{entity['entity_name']}** | {entity['ecosystem_role']} | {len(matched)} matching"):
            st.markdown(f"**Ecosystem Role:** {entity['ecosystem_role']}")

            if pd.notna(entity.get('brief_description')):
                st.markdown(f"**Description:**")
                st.write(entity['brief_description'])

            st.markdown(f"**Matching Activities:** {', '.join(matched)}")

            if entity_activities:
                other_activities = [a for a in entity_activities if a not in matched]
                if other_activities:
                    st.markdown(f"**Other Activities:** {', '.join(other_activities)}")

            if pd.notna(entity.get('latitude')) and pd.notna(entity.get('longitude')):
                st.markdown(f"**Location:** Lat {entity['latitude']:.6f}, Lon {entity['longitude']:.6f}")

    if len(matching_entities) > 50:
        st.info(f"Showing first 50 of {len(matching_entities)} matching entities.")


def show_capabilities_browser(df):
    """Browse capabilities offered by entities"""
    st.subheader("Browse Capabilities")

    # Definition for Capabilities
    with st.expander("What is a Capability?", expanded=False):
        st.markdown("""
        A **Capability** is what an organization can offer or provide to others in the ecosystem.
        It represents the skills, resources, technologies, or services that an entity has available
        and could share with potential partners.

        *Examples: recycling infrastructure, repair expertise, collection logistics,
        R&D facilities, training programs.*
        """)

    st.markdown("""
    Browse what organizations in this ecosystem can offer. Select a category to find entities
    with matching capabilities for potential collaboration.
    """)

    # Extract all capabilities and their categories
    all_capabilities = []
    for _, row in df.iterrows():
        caps = row.get('capabilities_parsed', [])
        if caps:
            for cap in caps:
                if isinstance(cap, dict):
                    all_capabilities.append({
                        'entity_name': row['entity_name'],
                        'ecosystem_role': row['ecosystem_role'],
                        'capability_name': cap.get('capability_name', 'Unknown'),
                        'description': cap.get('description', ''),
                        'category': cap.get('category', 'Uncategorized'),
                        'brief_description': row.get('brief_description', ''),
                    })

    if not all_capabilities:
        st.info("No capabilities data available for this ecosystem.")
        return

    caps_df = pd.DataFrame(all_capabilities)

    # Get unique categories
    categories = sorted(caps_df['category'].unique())

    # Category filter
    selected_category = st.selectbox(
        "Select Category",
        options=["-- All categories --"] + categories,
        help="Filter capabilities by category"
    )

    st.markdown("---")

    # Filter by category
    if selected_category != "-- All categories --":
        filtered_caps = caps_df[caps_df['category'] == selected_category]
    else:
        filtered_caps = caps_df

    # Show metrics
    col1, col2 = st.columns(2)
    col1.metric("Capabilities", len(filtered_caps))
    col2.metric("Entities", filtered_caps['entity_name'].nunique())

    if len(filtered_caps) == 0:
        st.info("No capabilities found for the selected category.")
        return

    # Group by entity and display
    for entity_name in filtered_caps['entity_name'].unique()[:50]:
        entity_caps = filtered_caps[filtered_caps['entity_name'] == entity_name]
        entity_role = entity_caps.iloc[0]['ecosystem_role']

        with st.expander(f"**{entity_name}** | {entity_role} | {len(entity_caps)} capabilities"):
            for _, cap in entity_caps.iterrows():
                st.markdown(f"**{cap['capability_name']}** ({cap['category']})")
                if cap['description']:
                    st.write(f"  {cap['description']}")

    if filtered_caps['entity_name'].nunique() > 50:
        st.info(f"Showing first 50 of {filtered_caps['entity_name'].nunique()} entities.")


def show_needs_browser(df):
    """Browse needs/requirements of entities"""
    st.subheader("Browse Needs")

    # Definition for Needs
    with st.expander("What is a Need?", expanded=False):
        st.markdown("""
        A **Need** is what an organization requires or is looking for from others in the ecosystem.
        It represents gaps, requirements, or resources that an entity is seeking through collaboration.

        *Examples: secondary raw materials, technical expertise, funding, partners for joint projects,
        access to specific markets.*
        """)

    st.markdown("""
    Browse what organizations in this ecosystem are looking for. Find entities whose needs
    match your capabilities for potential collaboration.
    """)

    # Extract all needs and their categories
    all_needs = []
    for _, row in df.iterrows():
        needs = row.get('needs_parsed', [])
        if needs:
            for need in needs:
                if isinstance(need, dict):
                    all_needs.append({
                        'entity_name': row['entity_name'],
                        'ecosystem_role': row['ecosystem_role'],
                        'need_name': need.get('need_name', 'Unknown'),
                        'description': need.get('description', ''),
                        'category': need.get('category', 'Uncategorized'),
                        'brief_description': row.get('brief_description', ''),
                    })

    if not all_needs:
        st.info("No needs data available for this ecosystem.")
        return

    needs_df = pd.DataFrame(all_needs)

    # Get unique categories
    categories = sorted(needs_df['category'].unique())

    # Category filter
    selected_category = st.selectbox(
        "Select Category",
        options=["-- All categories --"] + categories,
        help="Filter needs by category",
        key="needs_category"
    )

    st.markdown("---")

    # Filter by category
    if selected_category != "-- All categories --":
        filtered_needs = needs_df[needs_df['category'] == selected_category]
    else:
        filtered_needs = needs_df

    # Show metrics
    col1, col2 = st.columns(2)
    col1.metric("Needs", len(filtered_needs))
    col2.metric("Entities", filtered_needs['entity_name'].nunique())

    if len(filtered_needs) == 0:
        st.info("No needs found for the selected category.")
        return

    # Group by entity and display
    for entity_name in filtered_needs['entity_name'].unique()[:50]:
        entity_needs = filtered_needs[filtered_needs['entity_name'] == entity_name]
        entity_role = entity_needs.iloc[0]['ecosystem_role']

        with st.expander(f"**{entity_name}** | {entity_role} | {len(entity_needs)} needs"):
            for _, need in entity_needs.iterrows():
                st.markdown(f"**{need['need_name']}** ({need['category']})")
                if need['description']:
                    st.write(f"  {need['description']}")

    if filtered_needs['entity_name'].nunique() > 50:
        st.info(f"Showing first 50 of {filtered_needs['entity_name'].nunique()} entities.")


def show_cluster_collaboration(clusters_df):
    """Show collaboration opportunities within clusters with entity lists"""
    st.subheader("Explore Clusters")

    # Cluster type filter
    cluster_type = st.selectbox(
        "Cluster Type",
        options=['All'] + list(clusters_df['cluster_type'].unique())
    )

    filtered_clusters = clusters_df if cluster_type == 'All' else clusters_df[clusters_df['cluster_type'] == cluster_type]

    st.metric("Clusters", len(filtered_clusters))

    # Cluster selection dropdown
    cluster_names = ['All Clusters'] + sorted(filtered_clusters['cluster_name'].tolist())
    selected_cluster = st.selectbox(
        "Select a cluster",
        options=cluster_names,
        help="Choose a cluster to view its details and entities"
    )

    # Display clusters
    display_count = 0
    for _, cluster in filtered_clusters.iterrows():
        try:
            entities = json.loads(cluster['entities']) if isinstance(cluster['entities'], str) else cluster['entities']
            entity_count = len(entities) if entities else 0
        except:
            entities = []
            entity_count = 0

        # Apply cluster selection filter
        if selected_cluster != 'All Clusters' and cluster['cluster_name'] != selected_cluster:
            continue

        display_count += 1

        # Limit display
        if display_count > 50:
            st.info("Showing first 50 clusters. Use search to find more specific clusters.")
            break

        cluster_color = {
            'capability': 'synergy-card',
            'activity': 'recommendation-card',
            'need': 'gap-card'
        }.get(cluster['cluster_type'], 'insight-card')

        # Use expander for cluster with entity list
        with st.expander(
            f"**{cluster['cluster_name']}** | {cluster['cluster_type'].title()} | {entity_count} entities | {cluster['confidence']:.0%} confidence",
            expanded=False
        ):
            st.markdown(f"**Description:**")
            st.write(cluster['description'])  # Full description, not truncated

            st.markdown(f"**Type:** {cluster['cluster_type'].title()}")
            st.markdown(f"**Confidence:** {cluster['confidence']:.0%}")

            # Display entity list
            if entities and len(entities) > 0:
                st.markdown(f"**Entities in this cluster ({len(entities)}):**")

                # For small clusters, show all as bullet list
                if len(entities) <= 20:
                    for entity in sorted(entities):
                        st.markdown(f"- {entity}")
                else:
                    # For large clusters, show in columns
                    st.markdown(f"*Showing {min(len(entities), 50)} of {len(entities)} entities*")

                    # Create searchable list for large clusters
                    entity_search = st.text_input(
                        f"Search within {cluster['cluster_name']}",
                        key=f"search_{cluster['cluster_id']}",
                        placeholder="Filter entities..."
                    )

                    display_entities = entities
                    if entity_search:
                        display_entities = [e for e in entities if entity_search.lower() in str(e).lower()]

                    # Display in columns for better readability
                    cols = st.columns(2)
                    for idx, entity in enumerate(sorted(display_entities[:50])):
                        cols[idx % 2].markdown(f"- {entity}")

                    if len(display_entities) > 50:
                        st.info(f"+ {len(display_entities) - 50} more entities. Use search to filter.")
            else:
                st.info("No entities in this cluster.")


if __name__ == "__main__":
    main()
