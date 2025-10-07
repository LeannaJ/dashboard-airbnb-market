# 🏠 Airbnb Market Analytics Dashboard

A comprehensive Streamlit dashboard for analyzing Airbnb market data across major US cities (Chicago, Houston, Philadelphia, and Washington DC).

## 📊 Features

- **Interactive Dashboard**: Real-time filtering and analysis of Airbnb market data
- **Multi-City Analysis**: Compare performance across 4 major US cities
- **Key Performance Indicators**: Revenue, occupancy rates, ratings, and profitability metrics
- **Data Visualizations**: 
  - Market trends over time
  - Property type distribution
  - Geographic analysis with maps
  - Performance correlation analysis
- **Strategic Insights**: Data-driven recommendations for Airbnb investment decisions

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Airbnb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run Streamlit_AirBnB.py
```

## 📁 Project Structure

```
Airbnb/
├── Streamlit_AirBnB.py          # Main Streamlit application
├── airbnb_Chicago.csv           # Chicago Airbnb data
├── airbnb_Houston.csv           # Houston Airbnb data
├── airbnb_Philadelphia.csv      # Philadelphia Airbnb data
├── airbnb_Washington.csv        # Washington DC Airbnb data
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Git ignore file
```

## 🎯 Dashboard Sections

### 1. Market Trends
- Quarterly listing count trends
- Annual price and occupancy rate analysis
- Revenue trends over time

### 2. Property Type Analysis
- Property type distribution (pie charts)
- Price distribution by bedrooms/bathrooms
- Property characteristics correlation matrix

### 3. Performance Metrics
- Top 10 profitability score properties
- Price distribution analysis
- Reviews vs profitability correlation
- Rating vs profitability correlation

### 4. Geographic Analysis
- Top revenue-generating neighborhoods
- Interactive property distribution maps
- Superhost ratio by neighborhood

## 🔧 Technologies Used

- **Python**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computing

## 📈 Key Insights

The dashboard provides insights into:
- Market volatility patterns across all cities
- Universal apartment dominance (38-58% market share)
- Strong correlation between occupancy rate and profitability
- Geographic concentration in top-performing neighborhoods
- Quality metrics as primary success factors

## 🚀 Deployment

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set the main file path to `Streamlit_AirBnB.py`
5. Deploy!

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run Streamlit_AirBnB.py
```

## 📊 Data Sources

- Synthetic Airbnb market data for 4 major US cities
- Historical data from 2016-2020
- Property characteristics, pricing, and performance metrics
- Geographic and demographic information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, please open an issue in the repository.

---

**Built with ❤️ using Python, Streamlit, and Plotly**
