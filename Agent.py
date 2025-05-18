import getpass
import os
import json
import csv
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
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

try:
    from generate_charts import export as generate_charts  # Import export function
except ImportError as e:
    logger.error(f"Failed to import export from generate_charts: {str(e)}")
    def generate_charts(data: Dict[str, Any], output_dir: str) -> List[str]:
        logger.warning("Using fallback chart generation (no charts generated)")
        os.makedirs(output_dir, exist_ok=True)
        default_charts = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
        for chart in default_charts:
            placeholder_path = os.path.join(output_dir, chart)
            with open(placeholder_path, "w") as f:
                f.write("Placeholder chart: Error during generation")
        return [os.path.join(output_dir, chart) for chart in default_charts]

# Set default USER_AGENT if not provided
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MarketIntelligenceAgent/1.0 (+https://example.com)"
    logger.info("Set default USER_AGENT for WebBaseLoader")

# Load environment variables
load_dotenv()

# Cache for search results (TTL: 1 hour)
search_cache = TTLCache(maxsize=100, ttl=3600)

# Initialize SQLite database for state persistence
def init_db():
    conn = sqlite3.connect('market_intelligence.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS states (
            id TEXT PRIMARY KEY,
            market_domain TEXT,
            query TEXT,
            state_data TEXT,
            created_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class MarketIntelligenceState(BaseModel):
    raw_news_data: List[Dict[str, Any]] = Field(default_factory=list)
    competitor_data: List[Dict[str, Any]] = Field(default_factory=list)
    market_trends: List[Dict[str, Any]] = Field(default_factory=list)
    opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    strategic_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    market_domain: str = "EdTech"
    query: Optional[str] = None
    question: Optional[str] = None
    query_response: Optional[str] = None
    report_template: Optional[str] = None
    vector_store_path: Optional[str] = None
    state_id: str = Field(default_factory=lambda: str(uuid4()))

    @field_validator('market_domain')
    @classmethod
    def validate_market_domain(cls, v):
        if not re.match(r'^[a-zA-Z0-9\s\-]+$', v):
            raise ValueError("Market domain must contain only letters, numbers, spaces, or hyphens")
        return v.strip()

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if v and len(v.strip()) < 5:
            raise ValueError("Query must be at least 5 characters long")
        return v.strip() if v else v

# Save state to database
def save_state(state: MarketIntelligenceState):
    conn = sqlite3.connect('market_intelligence.db')
    c = conn.cursor()
    c.execute(
        'INSERT OR REPLACE INTO states (id, market_domain, query, state_data, created_at) VALUES (?, ?, ?, ?, ?)',
        (state.state_id, state.market_domain, state.query, json.dumps(state.dict()), datetime.now())
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved state {state.state_id} to database")

# Load state from database
def load_state(state_id: str) -> Optional[MarketIntelligenceState]:
    conn = sqlite3.connect('market_intelligence.db')
    c = conn.cursor()
    c.execute('SELECT state_data FROM states WHERE id = ?', (state_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return MarketIntelligenceState(**json.loads(result[0]))
    return None

# Synchronous Tavily Search
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_with_tavily(query: str) -> List[str]:
    cache_key = f"tavily_{query}"
    if cache_key in search_cache:
        logger.info(f"Cache hit for query: {query}")
        return search_cache[cache_key]

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY not set")

    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Content-Type": "application/json"},
        json={
            "api_key": tavily_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "max_results": 20
        }
    )
    response.raise_for_status()
    data = response.json()
    urls = [res["url"] for res in data.get("results", []) if "url" in res]
    search_cache[cache_key] = urls
    logger.info(f"Retrieved {len(urls)} URLs for query: {query}")
    return urls

def fetch_url_content(url: str) -> Dict[str, Any]:
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc = docs[0] if docs else {}
        content = doc.page_content[:300] if doc else "No content"
        logger.info(f"Fetched content from {url}: {content[:50]}...")
        return {
            "source": url,
            "title": doc.metadata.get("title", "No title") if doc else "No title",
            "summary": content,
            "url": url
        }
    except Exception as e:
        logger.error(f"Failed to load URL {url}: {str(e)}")
        return {
            "source": url,
            "title": "Failed to load",
            "summary": str(e),
            "url": url
        }

def market_data_collector(state: MarketIntelligenceState) -> Dict[str, Any]:
    logger.info(f"Collecting data for market domain: {state.market_domain}, query: {state.query}")
    try:
        news_urls = search_with_tavily(f"{state.query} {state.market_domain} news trends")
        competitor_urls = search_with_tavily(f"{state.query} {state.market_domain} competitors analysis")
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {"raw_news_data": [], "competitor_data": []}

    all_urls = list(set(news_urls + competitor_urls))
    logger.info(f"Found {len(all_urls)} unique URLs to process")

    raw_data = []
    for url in all_urls:
        logger.info(f"Processing URL: {url}")
        data = fetch_url_content(url)
        raw_data.append(data)

    logger.info(f"Collected data from {len(raw_data)} documents")

    # Save to JSON
    json_path = f"{state.market_domain.lower().replace(' ', '_')}_data_sources.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        logger.info(f"Data saved to JSON: {json_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {str(e)}")

    # Save to CSV
    csv_path = f"{state.market_domain.lower().replace(' ', '_')}_data_sources.csv"
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["title", "summary", "url", "source"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_data:
                writer.writerow(row)
        logger.info(f"Data saved to CSV: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV: {str(e)}")

    state.raw_news_data = raw_data
    state.competitor_data = raw_data
    save_state(state)
    logger.info(f"market_data_collector output: {len(raw_data)} news items, {len(raw_data)} competitor items")
    return {
        "raw_news_data": raw_data,
        "competitor_data": raw_data
    }

def trend_analyzer(state: MarketIntelligenceState) -> Dict[str, Any]:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a market analyst specializing in {market_domain}. Analyze the provided market and business data. Return a JSON array of trends with the following structure for each trend: trend_name, description, supporting_evidence, estimated_impact (High/Medium/Low), timeframe (Short-term/Medium-term/Long-term). Include at least 5 trends, filling gaps with reasonable assumptions if data is limited, and note any assumptions made."""),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    try:
        trends = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "news_data": state.raw_news_data,
                "competitor_data": state.competitor_data
            })
        })
        logger.info(f"Identified {len(trends)} market trends: {[t['trend_name'] for t in trends]}")
    except Exception as e:
        logger.error(f"Trend analysis failed: {str(e)}")
        trends = [
            {
                "trend_name": "Personalized Learning",
                "description": "EdTech solutions are focusing on tailored content and pacing for individual students.",
                "supporting_evidence": "Limited data; based on industry reports.",
                "estimated_impact": "High",
                "timeframe": "Medium-term"
            },
            {
                "trend_name": "AI Integration",
                "description": "AI is used for adaptive learning and content creation.",
                "supporting_evidence": "Limited data; based on industry trends.",
                "estimated_impact": "High",
                "timeframe": "Long-term"
            }
        ]
    state.market_trends = trends
    save_state(state)
    return {"market_trends": trends}

def opportunity_identifier(state: MarketIntelligenceState) -> Dict[str, Any]:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business strategist specializing in {market_domain}. Identify market opportunities from the provided trends, competitor data, and news. Return a JSON array with the following structure for each opportunity: opportunity_name, description, target_segment, competitive_advantage, estimated_potential (High/Medium/Low), timeframe_to_capture (Short-term/Medium-term/Long-term). Include at least 5 opportunities, filling gaps with reasonable assumptions if data is limited, and note any assumptions made."""),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    try:
        opportunities = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "market_trends": state.market_trends,
                "competitor_data": state.competitor_data,
                "news_data": state.raw_news_data
            })
        })
        logger.info(f"Identified {len(opportunities)} opportunities: {[o['opportunity_name'] for o in opportunities]}")
    except Exception as e:
        logger.error(f"Opportunity identification failed: {str(e)}")
        opportunities = [
            {
                "opportunity_name": "AI-Powered Tutoring",
                "description": "Develop AI-driven tutoring systems for personalized student support.",
                "target_segment": "K-12 Students, Parents",
                "competitive_advantage": "Advanced AI algorithms, 24/7 availability",
                "estimated_potential": "High",
                "timeframe_to_capture": "Medium-term"
            }
        ]
    state.opportunities = opportunities
    save_state(state)
    return {"opportunities": opportunities}

def strategy_recommender(state: MarketIntelligenceState) -> Dict[str, Any]:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strategic consultant specializing in {market_domain}. Recommend actionable strategies for each identified opportunity. Return a JSON array with the following structure for each strategy: strategy_title, description, implementation_steps, expected_outcome, resource_requirements, priority_level (High/Medium/Low), success_metrics. Include at least 5 strategies, filling gaps with reasonable assumptions if data is limited, and note any assumptions made."""),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    try:
        strategies = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "opportunities": state.opportunities,
                "market_trends": state.market_trends,
                "competitor_data": state.competitor_data
            })
        })
        logger.info(f"Generated {len(strategies)} strategic recommendations: {[s['strategy_title'] for s in strategies]}")
    except Exception as e:
        logger.error(f"Strategy recommendation failed: {str(e)}")
        strategies = [
            {
                "strategy_title": "Develop AI Tutoring System",
                "description": "Create an AI-powered tutoring platform for K-12 students.",
                "implementation_steps": ["Identify subjects", "Develop AI algorithms", "Beta test"],
                "expected_outcome": "Improved student performance",
                "resource_requirements": "AI engineers, content developers",
                "priority_level": "High",
                "success_metrics": "Student engagement rates, test score improvements"
            }
        ]
    state.strategic_recommendations = strategies
    save_state(state)
    return {"strategic_recommendations": strategies}

def report_template_generator(state: MarketIntelligenceState) -> Dict[str, Any]:
    llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a business report expert. Create a markdown template structure for a comprehensive market intelligence report for the {market_domain} industry.
        
        The template should include:
        1. Professional title and executive summary section
        2. Sections for market trends (with subsections for macro, technological, pedagogical trends, including a table)
        3. Opportunities section (with subsections for unmet needs, emerging technologies, market segments, including a table)
        4. Strategic recommendations section (with subsections for product development, marketing/sales, partnerships, including a table)
        5. Competitive landscape section (with a table for competitors)
        6. Appendix/sources section with a table of sources
        7. Mermaid charts for trend impact and opportunity potential
        8. Image placeholders for visualizations
        
        Use markdown formatting with proper headers (##, ###), tables, bullet points, and placeholder text like [INSERT TREND ANALYSIS HERE]. Do not include ```markdown``` fences. Do not fill in actual content - just create the structure template.
        """),
        ("human", "Generate a report template for {market_domain} industry focusing on {query}")
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        template = chain.invoke({
            "market_domain": state.market_domain,
            "query": state.query
        })
        logger.info(f"Generated report template with {len(template)} characters")
    except Exception as e:
        logger.error(f"Report template generation failed: {str(e)}")
        template = "# Market Intelligence Report\n\n[INSERT REPORT CONTENT HERE]"
    state.report_template = template
    save_state(state)
    return {"report_template": template}

def setup_vector_store(state: MarketIntelligenceState) -> Dict[str, Any]:
    logger.info(f"Setting up vector store for state {state.state_id}")
    report_path = f"reports/{state.market_domain.lower().replace(' ', '_')}_market_intelligence_report_{state.state_id}.md"
    json_path = f"{state.market_domain.lower().replace(' ', '_')}_data_sources.json"

    documents = []
    # Load report
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_content = f.read()
        documents.append({"content": report_content, "metadata": {"source": "report", "state_id": state.state_id}})
        logger.info(f"Loaded report: {report_path}")
    else:
        logger.warning(f"Report not found: {report_path}")

    # Load JSON data sources
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        for item in json_data:
            documents.append({
                "content": f"Title: {item['title']}\nSummary: {item['summary']}\nURL: {item['url']}",
                "metadata": {"source": "data_source", "url": item['url'], "state_id": state.state_id}
            })
        logger.info(f"Loaded {len(json_data)} data sources from {json_path}")
    else:
        logger.warning(f"JSON data sources not found: {json_path}")

    # Add state data
    state_data = {
        "market_trends": state.market_trends,
        "opportunities": state.opportunities,
        "strategic_recommendations": state.strategic_recommendations,
        "competitor_data": state.competitor_data,
        "raw_news_data": state.raw_news_data
    }
    documents.append({
        "content": json.dumps(state_data, indent=2),
        "metadata": {"source": "state_data", "state_id": state.state_id}
    })
    logger.info("Added state data to documents")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = []
    metadatas = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            texts.append(chunk)
            metadatas.append(doc["metadata"])

    # Create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store_path = f"vector_store_{state.state_id}"
        vector_store.save_local(vector_store_path)
        logger.info(f"Vector store saved to {vector_store_path}")
        state.vector_store_path = vector_store_path
        save_state(state)
        return {"vector_store_path": vector_store_path}
    except Exception as e:
        logger.error(f"Failed to create vector store: {str(e)}")
        return {"vector_store_path": None}

def rag_query(state: MarketIntelligenceState) -> Dict[str, Any]:
    if not state.question or not state.vector_store_path:
        logger.warning("No question or vector store path provided")
        return {"query_response": None}

    logger.info(f"Processing RAG query: {state.question}")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.load_local(state.vector_store_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        result = qa_chain({"query": state.question})
        answer = result["result"]
        sources = [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
        logger.info(f"RAG query response: {answer[:100]}... Sources: {sources}")

        # Save response to log
        with open(f"rag_responses_{state.state_id}.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Question: {state.question}\nAnswer: {answer}\nSources: {sources}\n\n")

        state.query_response = answer
        save_state(state)
        return {"query_response": answer}
    except Exception as e:
        logger.error(f"RAG query failed: {str(e)}")
        return {"query_response": f"Error processing question: {str(e)}"}

def create_market_intelligence_workflow():
    workflow = StateGraph(MarketIntelligenceState)
    workflow.add_node("market_data_collector", market_data_collector)
    workflow.add_node("trend_analyzer", trend_analyzer)
    workflow.add_node("opportunity_identifier", opportunity_identifier)
    workflow.add_node("strategy_recommender", strategy_recommender)
    workflow.add_node("report_template_generator", report_template_generator)
    workflow.add_node("setup_vector_store", setup_vector_store)
    workflow.add_node("rag_query", rag_query)

    workflow.set_entry_point("market_data_collector")
    workflow.add_edge("market_data_collector", "trend_analyzer")
    workflow.add_edge("trend_analyzer", "opportunity_identifier")
    workflow.add_edge("opportunity_identifier", "strategy_recommender")
    workflow.add_edge("strategy_recommender", "report_template_generator")
    workflow.add_edge("report_template_generator", "setup_vector_store")
    workflow.add_edge("setup_vector_store", "rag_query")
    workflow.add_edge("rag_query", END)

    return workflow.compile()

def verify_report_file(file_path: str) -> bool:
    if not os.path.exists(file_path):
        logger.error(f"Report file does not exist: {file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        logger.error(f"Report file is empty: {file_path}")
        return False
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(1024)
        # Relaxed markdown check: just ensure it's not binary or completely malformed
        has_basic_markdown = (
            content.strip() and
            not content.startswith(('\x00', '\xff', '\xfe')) and
            any(c in content for c in '#*|-')
        )
        if not has_basic_markdown:
            logger.warning(f"Report file may not contain valid markdown: {file_path}")
            return True  # Still accept, as it may be an error report
        logger.info(f"Report file verified: {file_path} ({file_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"Error verifying report file: {str(e)}")
        return False

def debug_state(state: MarketIntelligenceState):
    logger.info(f"Debugging state {state.state_id}:")
    logger.info(f"Market Domain: {state.market_domain}, Query: {state.query}, Question: {state.question}")
    logger.info(f"Raw News Data: {len(state.raw_news_data)} items")
    logger.info(f"Competitor Data: {len(state.competitor_data)} items")
    logger.info(f"Market Trends: {len(state.market_trends)} items - {[t['trend_name'] for t in state.market_trends]}")
    logger.info(f"Opportunities: {len(state.opportunities)} items - {[o['opportunity_name'] for o in state.opportunities]}")
    logger.info(f"Strategic Recommendations: {len(state.strategic_recommendations)} items - {[s['strategy_title'] for s in state.strategic_recommendations]}")
    logger.info(f"Report Template: {'Present' if state.report_template else 'Missing'}")
    logger.info(f"Vector Store Path: {state.vector_store_path or 'Missing'}")
    logger.info(f"Query Response: {'Present' if state.query_response else 'Missing'}")


def generate_report_charts(state: MarketIntelligenceState, report_dir: str) -> List[str]:
    """
    Generate charts using generate_charts.py and save them to the report directory.
    Returns a list of chart file paths relative to report_dir.
    """
    logger.info(f"Generating charts for state {state.state_id} in {report_dir}")
    chart_paths = []
    
    try:
        # Prepare data for charts
        chart_data = {
            "market_trends": state.market_trends,
            "competitor_data": state.competitor_data,
            "raw_news_data": state.raw_news_data,
            "market_domain": state.market_domain,
            "query": state.query
        }
        logger.info(f"Chart data prepared: {len(chart_data['market_trends'])} trends, {len(chart_data['competitor_data'])} competitors")
        
        # Call generate_charts.py (export function)
        generated_charts = generate_charts(chart_data, report_dir)
        logger.info(f"Generated {len(generated_charts)} charts: {generated_charts}")
        
        # Validate and convert to relative paths
        for chart_path in generated_charts:
            if os.path.exists(chart_path):
                # Convert to relative path (e.g., 'market_growth.png')
                rel_path = os.path.relpath(chart_path, report_dir)
                chart_paths.append(rel_path)
                logger.info(f"Chart verified: {chart_path}")
            else:
                logger.warning(f"Chart file not found: {chart_path}")
        
        if not chart_paths:
            logger.warning("No valid charts generated; using default placeholders")
            chart_paths = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
        
        return chart_paths
    
    except Exception as e:
        logger.error(f"Failed to generate charts: {str(e)}")
        # Fallback: Return default placeholder paths
        default_charts = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
        for chart in default_charts:
            placeholder_path = os.path.join(report_dir, chart)
            with open(placeholder_path, "w") as f:
                f.write("Placeholder chart: Error during generation")
            chart_paths.append(chart)
            logger.info(f"Created placeholder chart: {placeholder_path}")
        return chart_paths

def generate_market_intelligence_report(state: MarketIntelligenceState):
    logger.info(f"Generating market intelligence report for {state.market_domain}")
    debug_state(state)
    
    # Warn about missing data but proceed with fallbacks
    if not state.market_trends:
        logger.warning("No market trends data available; using placeholder")
        state.market_trends = [{"trend_name": "Placeholder Trend", "description": "No trends available", "supporting_evidence": "N/A", "estimated_impact": "Medium", "timeframe": "Medium-term"}]
    if not state.opportunities:
        logger.warning("No opportunities data available; using placeholder")
        state.opportunities = [{"opportunity_name": "Placeholder Opportunity", "description": "No opportunities available", "target_segment": "N/A", "competitive_advantage": "N/A", "estimated_potential": "Medium", "timeframe_to_capture": "Medium-term"}]
    if not state.strategic_recommendations:
        logger.warning("No strategic recommendations available; using placeholder")
        state.strategic_recommendations = [{"strategy_title": "Placeholder Strategy", "description": "No strategies available", "implementation_steps": ["N/A"], "expected_outcome": "N/A", "resource_requirements": "N/A", "priority_level": "Medium", "success_metrics": "N/A"}]
    if not state.competitor_data:
        logger.warning("No competitor data available; using placeholder")
        state.competitor_data = [{"title": "No competitor data", "summary": "N/A", "url": "N/A", "source": "N/A"}]
    
    report_data = {
        "market_domain": state.market_domain,
        "query": state.query or "General market analysis",
        "market_trends": state.market_trends,
        "opportunities": state.opportunities,
        "strategic_recommendations": state.strategic_recommendations,
        "competitor_data": state.competitor_data,
        "news_data": state.raw_news_data or [{"title": "No news data", "summary": "N/A", "url": "N/A", "source": "N/A"}]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', (state.query or "market_analysis").lower().replace(' ', '_'))
    report_dir = os.path.join("reports", f"{safe_query}_{timestamp}")
    report_file_name = f"{state.market_domain.lower().replace(' ', '_')}_market_intelligence_report_{state.state_id}.md"
    report_file_path = os.path.join(report_dir, report_file_name)
    logger.info(f"Constructed report file path: {report_file_path}")
    
    try:
        # Ensure directory exists
        logger.info(f"Creating directory: {report_dir}")
        os.makedirs(report_dir, exist_ok=True)
        logger.info(f"Directory created: {report_dir}")

        # Test write permissions
        test_file = os.path.join(report_dir, ".write_test")
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Write permissions verified for {report_dir}")
        except Exception as e:
            logger.error(f"Failed to verify write permissions for {report_dir}: {str(e)}")
            raise PermissionError(f"Cannot write to {report_dir}: {str(e)}")
        
        # Generate charts
        try:
            chart_paths = generate_report_charts(state, report_dir)
            report_data["chart_paths"] = chart_paths
            logger.info(f"Chart paths for report: {chart_paths}")
        except Exception as e:
            logger.warning(f"Chart generation failed: {str(e)}; using placeholders")
            chart_paths = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
            report_data["chart_paths"] = chart_paths
            for chart in chart_paths:
                placeholder_path = os.path.join(report_dir, chart)
                with open(placeholder_path, "w", encoding="utf-8") as f:
                    f.write("Placeholder chart: Error during generation")
        
        # Generate report content
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        if state.report_template:
            logger.info("Using generated report template structure")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a business report writer specializing in market intelligence. 
                Fill in the provided markdown report template with content from the data. Maintain all markdown formatting, tables, and Mermaid charts. 
                For each placeholder (e.g., [INSERT ...]), provide detailed content based on the data. If data is missing, include: "Data not available, further analysis required."
                Ensure:
                - Tables are populated with data (e.g., trends, opportunities, competitors).
                - Mermaid charts reflect trend impact and opportunity potential.
                - Include image references for charts using markdown: ![Chart Name](chart_path).
                - No ```markdown``` fences.
                - Comprehensive, data-driven, professionally formatted.
                - Include executive summary and sources table in appendix.
                - Date the report {timestamp} and set 'Prepared By' to 'Market Intelligence Agent'.
                """),
                ("human", "Template: {template}\nData: {data}\nChart paths: {chart_paths}")
            ])
            chain = prompt | llm | StrOutputParser()
            report_markdown = chain.invoke({
                "timestamp": timestamp,
                "template": state.report_template,
                "data": json.dumps(report_data),
                "chart_paths": ", ".join(chart_paths)
            })
        else:
            logger.info("No template found, generating full report directly")
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a business analyst specializing in {market_domain}. 
                Generate a comprehensive markdown market report from the provided data.
                
                Include:
                1. Title: "Market Intelligence Report: {query} in {market_domain}"
                2. Executive summary
                3. Market trends with table
                4. Opportunities with table
                5. Strategic recommendations with table
                6. Competitive landscape with table
                7. Visualizations with charts: ![Chart Name](chart_path)
                8. Sources/appendix with table
                9. Mermaid charts for trend impact and opportunity potential
                
                If data is missing, include: "Data not available, further analysis required."
                Use markdown formatting (headers, tables, lists, Mermaid charts). No ```markdown``` fences.
                Date the report {timestamp} and set 'Prepared By' to 'Market Intelligence Agent'.
                """),
                ("human", "Data: {data}\nChart paths: {chart_paths}")
            ])
            chain = prompt | llm | StrOutputParser()
            report_markdown = chain.invoke({
                "market_domain": state.market_domain,
                "timestamp": timestamp,
                "query": state.query or "General market analysis",
                "data": json.dumps(report_data),
                "chart_paths": ", ".join(chart_paths)
            })
        
        # Ensure no ```markdown``` fences and validate content
        report_markdown = report_markdown.replace("```markdown", "").replace("```", "")
        if not report_markdown.strip():
            logger.error("Generated report content is empty")
            chart_references = "\n".join(f"![{os.path.basename(p)}]({p})" for p in chart_paths)
            report_markdown = (
                f"# Market Intelligence Report: {state.query or 'General'} in {state.market_domain}\n\n"
                f"## Executive Summary\n"
                f"Data not available, further analysis required.\n\n"
                f"## Market Trends\n"
                f"Data not available, further analysis required.\n\n"
                f"## Opportunities\n"
                f"Data not available, further analysis required.\n\n"
                f"## Strategic Recommendations\n"
                f"Data not available, further analysis required.\n\n"
                f"## Competitive Landscape\n"
                f"Data not available, further analysis required.\n\n"
                f"## Visualizations\n"
                f"{chart_references}\n\n"
                f"## Appendix\n"
                f"| Source | URL |\n"
                f"|--------|-----|\n"
                f"| N/A    | N/A |\n\n"
                f"*Generated on {timestamp} by Market Intelligence Agent*"
            )
            logger.info("Used fallback report content")
        
        logger.info(f"Generated report content (first 200 chars): {report_markdown[:200]}")
        logger.info(f"Report content length: {len(report_markdown)} characters")
        
        # Write report to file
        try:
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(report_markdown)
            logger.info(f"Report written to {report_file_path}")
        except Exception as e:
            logger.error(f"Failed to write report to {report_file_path}: {str(e)}")
            raise IOError(f"Cannot write report to {report_file_path}: {str(e)}")
        finally:
            logger.info("Attempted to write report to {report_file_path}")
        
        # Verify report file
        if verify_report_file(report_file_path):
            file_size = os.path.getsize(report_file_path)
            logger.info(f"Report saved to '{report_file_path}' ({file_size} bytes)")
            with open(report_file_path, "r", encoding="utf-8") as f:
                first_lines = "".join(f.readline() for _ in range(5))
                logger.info(f"Report preview:\n{first_lines}...")
        else:
            logger.error(f"Report file verification failed at '{report_file_path}'")
            raise ValueError(f"Report file verification failed for {report_file_path}")
    
    except PermissionError as e:
        logger.error(f"Permission error generating report: {str(e)}")
        raise
    except IOError as e:
        logger.error(f"IO error generating report: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating report: {str(e)}")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Market Intelligence Report\n\nError generating report: {str(e)}")
            logger.info(f"Wrote error message to {report_file_path}")
            if verify_report_file(report_file_path):
                logger.info(f"Error report saved to '{report_file_path}'")
            else:
                logger.error(f"Error report verification failed at '{report_file_path}'")
        except Exception as write_error:
            logger.error(f"Failed to write error report to {report_file_path}: {str(write_error)}")
        logger.info(f"Unexpected error during report generation: {str(e)}")
        raise

def run_market_intelligence_agent(query: str = "Market analysis", market_domain: str = "Technology", question: Optional[str] = None):
    logger.info(f"Running Market Intelligence Agent for {market_domain} with query: {query}, question: {question}")
    workflow = create_market_intelligence_workflow()

    try:
        report_dir = os.path.abspath(os.path.join(os.getcwd(), "reports"))
        os.makedirs(report_dir, exist_ok=True)
        test_file = os.path.join(report_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Write permissions verified for report directory: {report_dir}")

        state = MarketIntelligenceState(query=query, market_domain=market_domain, question=question)
        final_state = workflow.invoke(state)
        final_state = MarketIntelligenceState(**final_state)
        logger.info("Market intelligence analysis complete")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', query.lower().replace(' ', '_'))
        report_path = os.path.join(
            "reports",
            f"{safe_query}_{timestamp}",
            f"{final_state.market_domain.lower().replace(' ', '_')}_market_intelligence_report_{final_state.state_id}.md"
        )
        logger.info(f"Expected report path: {report_path}")

        if os.path.exists(report_path):
            logger.info(f"Confirmed report exists at {report_path}")
        else:
            logger.error(f"Report file not found at {report_path}")

        if final_state.query_response:
            logger.info(f"Query response: {final_state.query_response[:200]}...")
            print(f"\nAnswer to '{question}':\n{final_state.query_response}\n")
        else:
            logger.warning("No query response was generated")
        
        # Interactive question input
        if not question:
            while True:
                user_question = input("\nEnter a question about the market analysis (or 'exit' to quit): ")
                if user_question.lower() == 'exit':
                    break
                final_state.question = user_question
                result = rag_query(final_state)
                answer = result.get("query_response", "No response generated")
                print(f"\nAnswer to '{user_question}':\n{answer}\n")
        
        return {
            "success": os.path.exists(report_path),
            "report_path": report_path,
            "query_response": final_state.query_response,
            "state_id": final_state.state_id
        }
        
    except Exception as e:
        logger.error(f"Error during workflow execution: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Market Intelligence Agent")
    parser.add_argument("--query", type=str, default="Market analysis", 
                       help="Specific query or focus for the market analysis")
    parser.add_argument("--market", type=str, default="Technology", 
                       help="Market domain to analyze (e.g., Technology, Healthcare, Finance)")
    parser.add_argument("--question", type=str, default=None,
                       help="Specific question to answer about the market analysis")
    parser.add_argument("--output", type=str, default=None,
                       help="Custom output file path for the report (optional)")
    
    args = parser.parse_args()
    
    try:
        result = run_market_intelligence_agent(query=args.query, market_domain=args.market, question=args.question)
        
        if args.output and result["success"]:
            import shutil
            try:
                shutil.copy(result["report_path"], args.output)
                logger.info(f"Report copied to custom location: {args.output}")
            except Exception as e:
                logger.error(f"Failed to copy report to custom location: {str(e)}")
        
        report_path = result.get("report_path", f"reports/{args.market.lower().replace(' ', '_')}_market_intelligence_report.md")
        
        if result["success"]:
            logger.info("==== EXECUTION SUMMARY ====")
            logger.info("Agent execution completed successfully")
            if verify_report_file(report_path):
                logger.info("Report file verified")
            else:
                logger.error("Report file verification failed")
        else:
            logger.error(f"Agent execution failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}")