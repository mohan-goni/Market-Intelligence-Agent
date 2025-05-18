import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from typing import Dict, List, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('market_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def export(data: Dict[str, Any] = None, output_dir: str = "reports") -> List[str]:
    """
    Generate charts for K-12 EdTech market intelligence report using scraped data.
    
    Args:
        data (Dict[str, Any], optional): Data from MarketIntelligenceState (e.g., market_trends, competitor_data).
        output_dir (str): Directory to save charts (default: 'reports').
    
    Returns:
        List[str]: List of absolute file paths to generated charts.
    """
    logger.info(f"Generating charts in {output_dir}")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Fallback data if none provided
    if data is None:
        data = {
            "market_trends": [
                {"trend_name": "Personalized Learning", "estimated_impact": "High"},
                {"trend_name": "AI Integration", "estimated_impact": "High"}
            ],
            "competitor_data": [
                {"title": "BYJU’S", "summary": "30% market share"},
                {"title": "IXL Learning", "summary": "20% market share"},
                {"title": "Duolingo", "summary": "15% market share"},
                {"title": "Others", "summary": "35% market share"}
            ],
            "raw_news_data": []
        }
    
    # --- Chart 1: Market Growth (Line Chart) ---
    years = np.arange(2025, 2034)
    # Extract CAGR and starting market size from data
    cagr = 0.245  # Default: 24.5%
    market_size = 50  # Default: $50B
    for item in data.get("market_trends", []) + data.get("raw_news_data", []):
        summary = item.get("summary", "").lower()
        if "cagr" in summary:
            try:
                # Extract number like "24.5%" or "0.245"
                cagr_match = re.search(r"(\d+\.?\d*)\s*(?:%|cagr)", summary)
                if cagr_match:
                    cagr = float(cagr_match.group(1)) / 100 if float(cagr_match.group(1)) > 1 else float(cagr_match.group(1))
            except:
                pass
        if "market size" in summary:
            try:
                # Extract number like "$50B" or "50 billion"
                size_match = re.search(r"(\d+\.?\d*)\s*(?:billion|\$b)", summary)
                if size_match:
                    market_size = float(size_match.group(1))
            except:
                pass
    
    growth = [market_size * (1 + cagr) ** t for t in range(len(years))]
    
    plt.figure(figsize=(8, 6))
    plt.plot(years, growth, 'b-', label=f'Projected Market Size (CAGR: {cagr*100:.1f}%)')
    plt.title('K-12 EdTech Market Growth (2025-2033)')
    plt.xlabel('Year')
    plt.ylabel('Market Size (Billions USD)')
    plt.grid(True)
    plt.legend()
    market_growth_path = os.path.join(output_dir, "market_growth.png")
    plt.savefig(market_growth_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {market_growth_path}")
    chart_paths.append(market_growth_path)
    
    # --- Chart 2: Competitor Market Share (Pie Chart) ---
    competitors = ['BYJU’S', 'IXL Learning', 'Duolingo', 'Others']
    shares = [30, 20, 15, 35]  # Default percentages
    # Extract competitor shares from competitor_data
    temp_competitors = []
    temp_shares = []
    for item in data.get("competitor_data", []):
        title = item.get("title", "")
        summary = item.get("summary", "").lower()
        if title and "market share" in summary:
            try:
                # Extract number like "30%" or "0.3"
                share_match = re.search(r"(\d+\.?\d*)\s*(?:%|market share)", summary)
                if share_match:
                    share = float(share_match.group(1)) if float(share_match.group(1)) <= 1 else float(share_match.group(1))
                    temp_competitors.append(title)
                    temp_shares.append(share)
            except:
                pass
    
    if temp_competitors and sum(temp_shares) > 0:
        # Normalize shares to sum to 100%
        total = sum(temp_shares)
        shares = [s * 100 / total for s in temp_shares]
        competitors = temp_competitors[:3] + ['Others'] if len(temp_competitors) > 3 else temp_competitors
        if len(competitors) > 3:
            shares = shares[:3] + [sum(shares[3:])]
    
    plt.figure(figsize=(8, 6))
    plt.pie(shares, labels=competitors, autopct='%1.1f%%', startangle=140, colors=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('K-12 EdTech Market Share (2025)')
    competitor_share_path = os.path.join(output_dir, "competitor_share.png")
    plt.savefig(competitor_share_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {competitor_share_path}")
    chart_paths.append(competitor_share_path)
    
    # --- Chart 3: Trend Impact (Bar Chart) ---
    trends = ['Personalized Learning', 'AI Integration', 'VR/AR', 'Cloud Computing', 'Cybersecurity']
    impacts = [0.9, 0.8, 0.6, 0.7, 0.5]  # Default impact scores
    # Extract trends and impacts from market_trends
    temp_trends = []
    temp_impacts = []
    for trend in data.get("market_trends", []):
        trend_name = trend.get("trend_name", "")
        impact = trend.get("estimated_impact", "").lower()
        if trend_name:
            # Map impact to score (0-1)
            impact_score = {'high': 0.9, 'medium': 0.6, 'low': 0.3}.get(impact, 0.5)
            temp_trends.append(trend_name)
            temp_impacts.append(impact_score)
    
    if temp_trends:
        trends = temp_trends[:5]
        impacts = temp_impacts[:5]
    
    plt.figure(figsize=(10, 6))
    plt.bar(trends, impacts, color='skyblue')
    plt.title('Impact of K-12 EdTech Trends')
    plt.xlabel('Trends')
    plt.ylabel('Estimated Impact (0-1)')
    plt.xticks(rotation=45)
    trend_impact_path = os.path.join(output_dir, "trend_impact.png")
    plt.savefig(trend_impact_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {trend_impact_path}")
    chart_paths.append(trend_impact_path)
    
    return chart_paths

if __name__ == "__main__":
    # Test the export function
    export()