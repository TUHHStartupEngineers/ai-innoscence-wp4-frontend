# AI-InnoScEnCE CE Ecosystem Explorer

Interactive visualization of Circular Economy ecosystems across Europe.
Part of the AI-InnoScEnCE project.

## Ecosystems

- **Hamburg, Germany**
- **Novi Sad, Serbia**
- **Cahul, Moldova**

## Features

- Interactive map with entity locations
- Entity browser with role-based filtering
- Partnership explorer with evidence
- CE activities taxonomy browser (120 activities in 10 categories)
- Capabilities and needs matching for collaboration opportunities
- AI-generated ecosystem insights

## Technology Stack

- **Streamlit** - Web application framework
- **PyDeck** - Interactive mapping
- **Plotly** - Data visualization
- **SQLite** - Database storage

## Deployment

Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).

<!-- Embedded on [https://ai-innoscence.eu/ecosystem](https://ai-innoscence.eu/ecosystem) -->

## Data Pipeline

The ecosystem data was extracted using:
1. **ScrapegraphAI** - AI-powered web scraping
2. **LLM Extraction** - Structured data extraction with Ollama (Model: Qwen 2.5:32b-instruct)
3. **Geocoding** - Google Maps API for location coordinates
4. **Analysis** - AI-generated insights and clustering

## Part of

**AI-InnoScEnCE Project**
AI-Empowered Innovation in Natural Science and Engineering for the Circular Economy.
[https://ai-innoscence.eu/](https://ai-innoscence.eu/)

## License

This project is part of the AI-InnoScEnCE project, a project funded by EIT HEI Initiative.