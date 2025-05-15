# EdTech Market Intelligence Agent

## Overview

This project implements an AI-powered EdTech Market Intelligence Agent using LangChain, LangGraph, and the Google Gemini API. The agent continuously analyzes the K-12 EdTech landscape to provide actionable business intelligence, helping startups make data-driven strategic decisions and identify emerging opportunities.

<p align="center">
  <img src="/api/placeholder/800/400" alt="EdTech Market Intelligence Agent Flow" />
</p>

## Features

- **Automated Market Data Collection**: Gathers information from EdTech news sources, competitor websites, and industry reports
- **Trend Analysis**: Identifies emerging patterns in teaching methodologies, technology adoption, and market shifts
- **Competitor Intelligence**: Creates detailed profiles of competitors including feature sets, pricing strategies, and market positioning
- **Opportunity Identification**: Discovers underserved segments, feature gaps, and strategic positioning opportunities
- **Strategic Recommendations**: Generates actionable insights for market positioning, product development, and go-to-market strategy
- **Interactive Query System**: Answers specific questions about market conditions and strategic options

## Architecture

The agent uses a modular, pipeline architecture built with LangGraph:

1. **Market Data Collector**: Gathers and organizes relevant market information
2. **Trend Analyzer**: Processes collected data to identify significant market trends
3. **Opportunity Identifier**: Discovers potential market gaps and strategic openings
4. **Strategy Recommender**: Generates actionable recommendations based on opportunities
5. **Query Engine**: Answers specific questions about the market analysis

## Prerequisites

- Python 3.8+
- Google Gemini API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/edtech-market-intelligence.git
   cd edtech-market-intelligence
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key_here"
   ```
   
   Or add it directly to the code in `market_intelligence_agent.py`:
   ```python
   os.environ["GOOGLE_API_KEY"] = "your_gemini_api_key_here"
   ```

## Usage

### Basic Usage

```python
from market_intelligence_agent import run_market_intelligence_agent, generate_market_intelligence_report

# Run the agent with an optional specific query
result = run_market_intelligence_agent(
    query="What are the most promising market opportunities in middle school math education?"
)

# Generate a comprehensive report
report = generate_market_intelligence_report(result)

# Save the report
with open("edtech_market_intelligence_report.md", "w") as f:
    f.write(report)
```

### Customizing Data Sources

To customize the news sources or add additional data collection methods, modify the `market_data_collector` function:

```python
def market_data_collector(state):
    # Add your custom data sources here
    custom_news_sources = [
        {"name": "Your EdTech Source", "url": "https://your-source.com/"}
    ]
    
    # Implement custom scraping logic
    # ...
```

### Extending the Agent

The agent can be extended with additional components by:

1. Creating a new component function following the pattern of existing ones
2. Adding it to the graph in `create_market_intelligence_workflow()`
3. Updating the state model to include new data fields

## Sample Output

The agent generates a comprehensive market intelligence report in Markdown format:

```markdown
# EdTech Market Intelligence Report

## Executive Summary
This report provides an analysis of current trends, opportunities, and strategic recommendations for the K-12 EdTech market...

## Key Market Trends
1. **AI-Powered Tutoring**: School districts are rapidly adopting AI-based tutoring solutions...
2. **Project-Based Assessment**: 45% growth in platforms that facilitate project-based evaluation...

## Competitor Analysis
...
```

## Customization Options

### Adding New Data Sources

Modify the `market_data_collector` function to include additional sources:
- Industry reports
- Social media monitoring
- School district technology plans
- App store reviews

### Enhancing Analysis Capabilities

Implement more specialized analysis components:
- Regional market analysis
- Funding pattern recognition
- Sentiment analysis of user reviews
- Educational policy impact assessment

## Limitations and Future Improvements

- Currently uses simulated data; can be enhanced with real web scraping
- Analysis depth is limited by Gemini API context window
- Future versions could include automated data refreshing on a schedule
- Potential to add visualization components for market mapping

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- LangChain for the core framework
- LangGraph for workflow orchestration
- Google Gemini for the large language model capabilities