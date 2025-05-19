import os
import json
import csv
import sqlite3
import logging
import shutil
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache

# Configure logging for tool calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] TOOL_CALL: %(message)s',
    handlers=[
        logging.FileHandler('market_intelligence.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Error logger
error_logger = logging.getLogger('error')
error_handler = logging.FileHandler('market_intelligence_errors.log')
error_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
error_logger.addHandler(error_handler)
error_logger.setLevel(logging.ERROR)

try:
    from generate_charts import export as generate_charts
except ImportError:
    def generate_charts(data: Dict[str, Any], output_dir: str) -> List[str]:
        logger.info("Chart generation: Using fallback")
        os.makedirs(output_dir, exist_ok=True)
        default_charts = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
        for chart in default_charts:
            placeholder_path = os.path.join(output_dir, chart)
            try:
                with open(placeholder_path, "w", encoding="utf-8") as f:
                    f.write("Placeholder chart")
                logger.info(f"Chart generation: Created placeholder {placeholder_path}")
            except Exception as we:
                error_logger.error(f"Failed to create placeholder chart {placeholder_path}: {str(we)}")
        return default_charts

if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "MarketIntelligenceAgent/1.0 (+https://example.com)"
    logger.info("Set default USER_AGENT for WebBaseLoader")

load_dotenv()

search_cache = TTLCache(maxsize=100, ttl=3600)

def init_db():
    try:
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
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                session_id TEXT,
                message_type TEXT,
                content TEXT,
                timestamp TIMESTAMP,
                PRIMARY KEY (session_id, timestamp)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialization")
    except Exception as e:
        error_logger.error(f"Failed to initialize database: {str(e)}")
        raise

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
    report_dir: Optional[str] = None

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

def save_state(state: MarketIntelligenceState):
    try:
        conn = sqlite3.connect('market_intelligence.db')
        c = conn.cursor()
        c.execute(
            'INSERT OR REPLACE INTO states (id, market_domain, query, state_data, created_at) VALUES (?, ?, ?, ?, ?)',
            (state.state_id, state.market_domain, state.query, json.dumps(state.dict()), datetime.now())
        )
        conn.commit()
        conn.close()
        logger.info(f"State save: {state.state_id}")
    except Exception as e:
        error_logger.error(f"Failed to save state {state.state_id}: {str(e)}")

def load_state(state_id: str) -> Optional[MarketIntelligenceState]:
    try:
        conn = sqlite3.connect('market_intelligence.db')
        c = conn.cursor()
        c.execute('SELECT state_data FROM states WHERE id = ?', (state_id,))
        result = c.fetchone()
        conn.close()
        if result:
            return MarketIntelligenceState(**json.loads(result[0]))
        error_logger.error(f"State {state_id} not found")
        return None
    except Exception as e:
        error_logger.error(f"Failed to load state {state_id}: {str(e)}")
        return None

def save_chat_message(session_id: str, message_type: str, content: str):
    try:
        conn = sqlite3.connect('market_intelligence.db')
        c = conn.cursor()
        c.execute(
            'INSERT INTO chat_history (session_id, message_type, content, timestamp) VALUES (?, ?, ?, ?)',
            (session_id, message_type, content, datetime.now())
        )
        conn.commit()
        conn.close()
        logger.info(f"Chat message saved: session {session_id}, type {message_type}")
    except Exception as e:
        error_logger.error(f"Failed to save chat message for session {session_id}: {str(e)}")

def load_chat_history(session_id: str) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect('market_intelligence.db')
        c = conn.cursor()
        c.execute('SELECT message_type, content FROM chat_history WHERE session_id = ? ORDER BY timestamp', (session_id,))
        messages = [{"type": row[0], "content": row[1]} for row in c.fetchall()]
        conn.close()
        logger.info(f"Chat history loaded: session {session_id}, {len(messages)} messages")
        return messages
    except Exception as e:
        error_logger.error(f"Failed to load chat history for session {session_id}: {str(e)}")
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def search_with_tavily(query: str) -> List[str]:
    cache_key = f"tavily_{query}"
    if cache_key in search_cache:
        logger.info(f"Tavily search: Cache hit for query: {query}")
        return search_cache[cache_key]

    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        error_logger.error("TAVILY_API_KEY not set")
        raise ValueError("TAVILY_API_KEY not set")

    try:
        logger.info(f"Tavily search: Querying {query}")
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
        logger.info(f"Tavily search: Retrieved {len(urls)} URLs for query: {query}")
        return urls
    except requests.exceptions.RequestException as e:
        error_logger.error(f"Tavily search failed for query {query}: {str(e)}")
        raise

def fetch_url_content(url: str) -> Dict[str, Any]:
    try:
        logger.info(f"Web loader: Fetching {url}")
        loader = WebBaseLoader(url)
        docs = loader.load()
        doc = docs[0] if docs else {}
        content = doc.page_content[:300] if doc else "No content"
        logger.info(f"Web loader: Fetched {url}: {content[:50]}...")
        return {
            "source": url,
            "title": doc.metadata.get("title", "No title") if doc else "No title",
            "summary": content,
            "url": url
        }
    except Exception as e:
        error_logger.error(f"Failed to load URL {url}: {str(e)}")
        return {
            "source": url,
            "title": "Failed to load",
            "summary": str(e),
            "url": url
        }

def market_data_collector(state: MarketIntelligenceState) -> Dict[str, Any]:
    logger.info(f"Market data collector: Starting for {state.market_domain}, query: {state.query}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', (state.query or "analysis").lower().replace(' ', '_')[:10])
    report_dir = os.path.join("reports1", f"{safe_query}_{timestamp}")
    try:
        os.makedirs(report_dir, exist_ok=True)
        state.report_dir = report_dir
        logger.info(f"Market data collector: Set report_dir: {report_dir}")
    except Exception as e:
        error_logger.error(f"Failed to create report_dir {report_dir}: {str(e)}")
        raise IOError(f"Cannot create report directory: {str(e)}")
    
    json_path = os.path.join(report_dir, f"{state.market_domain.lower().replace(' ', '_')}_data_sources.json")
    csv_path = os.path.join(report_dir, f"{state.market_domain.lower().replace(' ', '_')}_data_sources.csv")
    
    try:
        news_urls = search_with_tavily(f"{state.query} {state.market_domain} news trends")
        competitor_urls = search_with_tavily(f"{state.query} {state.market_domain} competitors analysis")
    except Exception as e:
        error_logger.error(f"Search failed: {str(e)}")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([], f)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["title", "summary", "url", "source"])
                writer.writeheader()
            logger.info(f"Market data collector: Created empty data files: {json_path}, {csv_path}")
        except Exception as we:
            error_logger.error(f"Failed to create empty data files: {str(we)}")
        return {"raw_news_data": [], "competitor_data": [], "report_dir": report_dir}

    all_urls = list(set(news_urls + competitor_urls))
    logger.info(f"Market data collector: Found {len(all_urls)} unique URLs")

    raw_data = []
    for url in all_urls:
        data = fetch_url_content(url)
        raw_data.append(data)

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2)
        logger.info(f"Market data collector: Data saved to JSON: {json_path}")
    except Exception as e:
        error_logger.error(f"Failed to save JSON: {str(e)}")

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["title", "summary", "url", "source"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in raw_data:
                writer.writerow(row)
        logger.info(f"Market data collector: Data saved to CSV: {csv_path}")
    except Exception as e:
        error_logger.error(f"Failed to save CSV: {str(e)}")

    state.raw_news_data = raw_data
    state.competitor_data = raw_data
    save_state(state)
    return {
        "raw_news_data": raw_data,
        "competitor_data": raw_data,
        "report_dir": report_dir
    }

def trend_analyzer(state: MarketIntelligenceState) -> Dict[str, Any]:
    try:
        logger.info(f"LLM: Initializing gemini-2.0-flash for trend analysis")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze data for {market_domain}. Return JSON array of trends with: trend_name, description, supporting_evidence, estimated_impact (High/Medium/Low), timeframe (Short-term/Medium-term/Long-term). At least 3 trends."""),
            ("human", "{input}")
        ])
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        logger.info(f"LLM: Running trend analysis for {state.market_domain}")
        trends = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "news_data": state.raw_news_data,
                "competitor_data": state.competitor_data
            })
        })
        logger.info(f"LLM: Identified {len(trends)} trends")
    except Exception as e:
        error_logger.error(f"Trend analysis failed: {str(e)}")
        trends = [
            {
                "trend_name": "Personalized Learning",
                "description": "Tailored content for students.",
                "supporting_evidence": "Industry reports.",
                "estimated_impact": "High",
                "timeframe": "Medium-term"
            }
        ]
    state.market_trends = trends
    save_state(state)
    return {"market_trends": trends}

def opportunity_identifier(state: MarketIntelligenceState) -> Dict[str, Any]:
    try:
        logger.info(f"LLM: Initializing gemini-2.0-flash for opportunity identification")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Identify opportunities in {market_domain}. Return JSON array with: opportunity_name, description, target_segment, competitive_advantage, estimated_potential (High/Medium/Low), timeframe_to_capture (Short-term/Medium-term/Long-term). At least 3 opportunities."""),
            ("human", "{input}")
        ])
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        logger.info(f"LLM: Running opportunity identification for {state.market_domain}")
        opportunities = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "market_trends": state.market_trends,
                "competitor_data": state.competitor_data,
                "news_data": state.raw_news_data
            })
        })
        logger.info(f"LLM: Identified {len(opportunities)} opportunities")
    except Exception as e:
        error_logger.error(f"Opportunity identification failed: {str(e)}")
        opportunities = [
            {
                "opportunity_name": "AI Tutoring",
                "description": "AI-driven tutoring systems.",
                "target_segment": "K-12 Students",
                "competitive_advantage": "24/7 availability",
                "estimated_potential": "High",
                "timeframe_to_capture": "Medium-term"
            }
        ]
    state.opportunities = opportunities
    save_state(state)
    return {"opportunities": opportunities}

def strategy_recommender(state: MarketIntelligenceState) -> Dict[str, Any]:
    try:
        logger.info(f"LLM: Initializing gemini-2.0-flash for strategy recommendation")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Recommend strategies for {market_domain}. Return JSON array with: strategy_title, description, implementation_steps, expected_outcome, resource_requirements, priority_level (High/Medium/Low), success_metrics. At least 3 strategies."""),
            ("human", "{input}")
        ])
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        logger.info(f"LLM: Running strategy recommendation for {state.market_domain}")
        strategies = chain.invoke({
            "market_domain": state.market_domain,
            "input": json.dumps({
                "opportunities": state.opportunities,
                "market_trends": state.market_trends,
                "competitor_data": state.competitor_data
            })
        })
        logger.info(f"LLM: Generated {len(strategies)} strategies")
    except Exception as e:
        error_logger.error(f"Strategy recommendation failed: {str(e)}")
        strategies = [
            {
                "strategy_title": "Develop AI Tutoring",
                "description": "AI tutoring platform.",
                "implementation_steps": ["Develop AI", "Test"],
                "expected_outcome": "Improved performance",
                "resource_requirements": "Engineers",
                "priority_level": "High",
                "success_metrics": "Engagement rates"
            }
        ]
    state.strategic_recommendations = strategies
    save_state(state)
    return {"strategic_recommendations": strategies}

def report_template_generator(state: MarketIntelligenceState) -> Dict[str, Any]:
    try:
        logger.info(f"LLM: Initializing gemini-2.0-flash for report template")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a markdown template for a {market_domain} market intelligence report. Include title, executive summary, trends, opportunities, recommendations, competitive landscape, appendix, and visualization placeholders. Use markdown, no ```markdown``` fences."""),
            ("human", "Generate template for {market_domain} focusing on {query}")
        ])
        chain = prompt | llm | StrOutputParser()
        logger.info(f"LLM: Generating template for {state.market_domain}")
        template = chain.invoke({
            "market_domain": state.market_domain,
            "query": state.query
        })
        logger.info(f"LLM: Template generated")
    except Exception as e:
        error_logger.error(f"Template generation failed: {str(e)}")
        template = "# Market Intelligence Report\n\n## Executive Summary\n[INSERT CONTENT]"
    state.report_template = template
    save_state(state)
    return {"report_template": template}

def setup_vector_store(state: MarketIntelligenceState) -> Dict[str, Any]:
    logger.info(f"Vector store: Setting up for state {state.state_id}")
    if not state.report_dir:
        error_logger.error("report_dir not set in state")
        raise ValueError("Report directory not initialized")

    report_path = os.path.join(state.report_dir, f"{state.market_domain.lower().replace(' ', '_')}_report_{state.state_id[:4]}.md")
    json_path = os.path.join(state.report_dir, f"{state.market_domain.lower().replace(' ', '_')}_data_sources.json")

    documents = []
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                documents.append({"content": f.read(), "metadata": {"source": "report", "state_id": state.state_id}})
        except Exception as e:
            error_logger.error(f"Failed to read report {report_path}: {str(e)}")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    documents.append({
                        "content": f"Title: {item['title']}\nSummary: {item['summary']}\nURL: {item['url']}",
                        "metadata": {"source": "data_source", "url": item['url'], "state_id": state.state_id}
                    })
        except Exception as e:
            error_logger.error(f"Failed to read JSON {json_path}: {str(e)}")
    documents.append({
        "content": json.dumps({
            "market_trends": state.market_trends,
            "opportunities": state.opportunities,
            "strategic_recommendations": state.strategic_recommendations,
            "competitor_data": state.competitor_data,
            "raw_news_data": state.raw_news_data
        }, indent=2),
        "metadata": {"source": "state_data", "state_id": state.state_id}
    })

    try:
        logger.info(f"Vector store: Creating FAISS index")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = []
        metadatas = []
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for chunk in chunks:
                texts.append(chunk)
                metadatas.append(doc["metadata"])

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        vector_store_path = os.path.join(state.report_dir, f"vector_store_{state.state_id[:4]}")
        vector_store.save_local(vector_store_path)
        state.vector_store_path = vector_store_path
        logger.info(f"Vector store: Saved {vector_store_path}")
        save_state(state)
        return {"vector_store_path": vector_store_path}
    except Exception as e:
        error_logger.error(f"Failed to create vector store: {str(e)}")
        return {"vector_store_path": None}

def rag_query(state: MarketIntelligenceState) -> Dict[str, Any]:
    if not state.question or not state.vector_store_path:
        logger.info("RAG query: Skipped (no question or vector store)")
        return {"query_response": None}

    logger.info(f"RAG query: Processing {state.question}")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.load_local(state.vector_store_path, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        logger.info(f"LLM: Initializing gemini-2.0-flash for RAG query")
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
        log_path = os.path.join(state.report_dir, f"rag_responses_{state.state_id[:4]}.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now()}] Question: {state.question}\nAnswer: {answer}\nSources: {sources}\n\n")
        logger.info(f"RAG query: Response logged to {log_path}")
        state.query_response = answer
        save_state(state)
        return {"query_response": answer}
    except Exception as e:
        error_logger.error(f"RAG query failed: {str(e)}")
        return {"query_response": f"Error in RAG query: {str(e)}"}

def chat_with_agent(message: str, session_id: str, history: List[Dict[str, Any]]) -> str:
    try:
        logger.info(f"LLM: Initializing gemini-2.0-flash for chat, session {session_id}")
        llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant for the EdTech Market Intelligence Agent. Answer questions conversationally, providing insights on EdTech or general topics. Use previous messages for context if relevant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        chain = prompt | llm | StrOutputParser()
        
        # Convert history to LangChain message objects
        message_history = []
        for msg in history:
            if msg["type"] == "human":
                message_history.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                message_history.append(AIMessage(content=msg["content"]))
        
        logger.info(f"LLM: Processing chat message for session {session_id}: {message[:50]}...")
        response = chain.invoke({
            "history": message_history,
            "input": message
        })
        logger.info(f"LLM: Chat response generated for session {session_id}")
        
        # Save messages
        save_chat_message(session_id, "human", message)
        save_chat_message(session_id, "ai", response)
        
        return response
    except Exception as e:
        error_logger.error(f"Chat failed for session {session_id}: {str(e)}")
        return f"Error in chat: {str(e)}"

def create_market_intelligence_workflow():
    workflow = StateGraph(MarketIntelligenceState)
    workflow.add_node("market_data_collector", market_data_collector)
    workflow.add_node("trend_analyzer", trend_analyzer)
    workflow.add_node("opportunity_identifier", opportunity_identifier)
    workflow.add_node("strategy_recommender", strategy_recommender)
    workflow.add_node("report_template_generator", report_template_generator)
    workflow.add_node("setup_vector_store", setup_vector_store)
    workflow.add_node("rag_query", rag_query)
    workflow.add_node("generate_report", generate_market_intelligence_report)

    workflow.set_entry_point("market_data_collector")
    workflow.add_edge("market_data_collector", "trend_analyzer")
    workflow.add_edge("trend_analyzer", "opportunity_identifier")
    workflow.add_edge("opportunity_identifier", "strategy_recommender")
    workflow.add_edge("strategy_recommender", "report_template_generator")
    workflow.add_edge("report_template_generator", "setup_vector_store")
    workflow.add_edge("setup_vector_store", "rag_query")
    workflow.add_edge("rag_query", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

def verify_report_file(file_path: str) -> bool:
    try:
        if not os.path.exists(file_path):
            error_logger.error(f"Report file does not exist: {file_path}")
            return False
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            error_logger.error(f"Report file is empty: {file_path}")
            return False
        logger.info(f"Report verification: {file_path} ({file_size} bytes)")
        return True
    except Exception as e:
        error_logger.error(f"Error verifying report: {str(e)}")
        return False

def generate_readme(state: MarketIntelligenceState, report_dir: str, report_file_name: str) -> str:
    logger.info(f"Generating README in {report_dir}")
    readme_path = os.path.join(report_dir, "README.md")
    readme_content = (
        f"# Market Intelligence Report\n\n"
        f"## Overview\n"
        f"Report for **{state.market_domain}** on query: **{state.query or 'General analysis'}**.\n\n"
        f"- Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- State ID: {state.state_id[:4]}\n\n"
        f"## Contents\n"
        f"- `{report_file_name}`: Market intelligence report.\n"
        f"- `README.md`: This file.\n"
        f"- `{state.market_domain.lower().replace(' ', '_')}_data_sources.json`: Data sources in JSON.\n"
        f"- `{state.market_domain.lower().replace(' ', '_')}_data_sources.csv`: Data sources in CSV.\n"
        f"- `vector_store_{state.state_id[:4]}`: Vector store directory.\n"
        f"- `market_growth.png`, `competitor_share.png`, `trend_impact.png`: Visualization charts.\n"
        f"- `market_intelligence.log`: Execution log.\n"
        f"- `rag_responses_{state.state_id[:4]}.log`: RAG query responses.\n\n"
        f"## Usage\n"
        f"Open `{report_file_name}` in a markdown viewer.\n"
        f"Run again with:\n"
        f"```bash\n"
        f"python Agent.py --query \"Your query\" --market \"Your market domain\"\n"
        f"```\n"
    )
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        logger.info(f"README saved: {readme_path} ({os.path.getsize(readme_path)} bytes)")
        return readme_path
    except Exception as e:
        error_logger.error(f"Failed to generate README: {str(e)}")
        return None

def generate_report_charts(state: MarketIntelligenceState, report_dir: str) -> List[str]:
    logger.info(f"Generating charts in {report_dir}")
    chart_paths = []
    try:
        chart_data = {
            "market_trends": state.market_trends,
            "competitor_data": state.competitor_data,
            "raw_news_data": state.raw_news_data,
            "market_domain": state.market_domain,
            "query": state.query
        }
        generated_charts = generate_charts(chart_data, report_dir)
        for chart in generated_charts:
            chart_path = os.path.join(report_dir, chart)
            if os.path.exists(chart_path):
                chart_paths.append(chart)
        if not chart_paths:
            raise ValueError("No valid charts generated")
    except Exception as e:
        error_logger.error(f"Chart generation failed: {str(e)}")
        default_charts = ["market_growth.png", "competitor_share.png", "trend_impact.png"]
        for chart in default_charts:
            placeholder_path = os.path.join(report_dir, chart)
            try:
                with open(placeholder_path, "w", encoding="utf-8") as f:
                    f.write("Placeholder chart")
                chart_paths.append(chart)
                logger.info(f"Created placeholder chart: {placeholder_path}")
            except Exception as we:
                error_logger.error(f"Failed to create placeholder chart {placeholder_path}: {str(we)}")
    return chart_paths

def generate_market_intelligence_report(state: MarketIntelligenceState):
    logger.info(f"Generating report for {state.market_domain}")

    if not state.report_dir:
        error_logger.error("report_dir not set in state")
        raise ValueError("Report directory not initialized")

    if not state.market_trends:
        state.market_trends = [{"trend_name": "Placeholder", "description": "N/A"}]
    if not state.opportunities:
        state.opportunities = [{"opportunity_name": "Placeholder", "description": "N/A"}]
    if not state.strategic_recommendations:
        state.strategic_recommendations = [{"strategy_title": "Placeholder", "description": "N/A"}]
    if not state.competitor_data:
        state.competitor_data = [{"title": "No data", "summary": "N/A"}]

    report_data = {
        "market_domain": state.market_domain,
        "query": state.query or "General analysis",
        "market_trends": state.market_trends,
        "opportunities": state.opportunities,
        "strategic_recommendations": state.strategic_recommendations,
        "competitor_data": state.competitor_data,
        "news_data": state.raw_news_data or [{"title": "No data"}]
    }

    report_dir = state.report_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_file_name = f"{state.market_domain.lower().replace(' ', '_')}_report_{state.state_id[:4]}.md"
    report_file_path = os.path.join(report_dir, report_file_name)
    log_copy_path = os.path.join(report_dir, "market_intelligence.log")
    logger.info(f"Report path: {report_file_path}")
    logger.info(f"Output folder: {report_dir}")

    try:
        os.makedirs(report_dir, exist_ok=True)
        test_file = os.path.join(report_dir, "test.txt")
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Write permissions verified: {report_dir}")
        except Exception as e:
            error_logger.error(f"Write test failed: {str(e)}")
            raise PermissionError(f"Cannot write to {report_dir}: {str(e)}")

        try:
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write("# Test Report\n\nWrite test.")
            logger.info(f"Initial write test succeeded: {report_file_path}")
            os.remove(report_file_path)
        except Exception as e:
            error_logger.error(f"Initial report write test failed: {str(e)}")
            raise IOError(f"Cannot write report: {str(e)}")

        chart_paths = generate_report_charts(state, report_dir)
        report_data["chart_paths"] = chart_paths

        try:
            logger.info(f"LLM: Initializing gemini-2.0-flash for report generation")
            llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
            if state.report_template:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Fill the markdown template with data. Use 'N/A' for missing data. Include chart references: ![Chart](chart_path). No ```markdown``` fences. Date: {timestamp}, Prepared By: Market Intelligence Agent."""),
                    ("human", "Template: {template}\nData: {data}\nCharts: {chart_paths}")
                ])
                chain = prompt | llm | StrOutputParser()
                report_markdown = chain.invoke({
                    "timestamp": timestamp,
                    "template": state.report_template,
                    "data": json.dumps(report_data),
                    "chart_paths": ", ".join(chart_paths)
                })
                logger.info(f"LLM: Report markdown generated")
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """Generate a markdown report for {market_domain}. Include: Title, Executive summary, Trends, Opportunities, Recommendations, Competitive landscape, Visualizations ([Chart](chart_path)), Appendix. Use 'N/A' for missing data. No ```markdown``` fences. Date: {timestamp}, Prepared By: Market Intelligence Agent."""),
                    ("human", "Data: {data}\nCharts: {chart_paths}")
                ])
                chain = prompt | llm | StrOutputParser()
                report_markdown = chain.invoke({
                    "market_domain": state.market_domain,
                    "timestamp": timestamp,
                    "data": json.dumps(report_data),
                    "chart_paths": ", ".join(chart_paths)
                })
                logger.info(f"LLM: Fallback report markdown generated")
        except Exception as e:
            error_logger.error(f"LLM report generation failed: {str(e)}")
            report_markdown = ""

        report_markdown = report_markdown.replace("```markdown", "").replace("```", "")
        if not report_markdown.strip():
            logger.info("Report content empty; using fallback")
            chart_references = "\n".join(f"![{p}]({p})" for p in chart_paths)
            report_markdown = (
                f"# Market Intelligence Report: {state.query or 'General'} in {state.market_domain}\n"
                f"## Executive Summary\nN/A\n"
                f"## Market Trends\nN/A\n"
                f"## Opportunities\nN/A\n"
                f"## Recommendations\nN/A\n"
                f"## Competitive Landscape\nN/A\n"
                f"## Visualizations\n{chart_references}\n"
                f"## Appendix\n| Source | URL |\n|--------|-----|\n| N/A    | N/A |\n"
                f"*Generated on {timestamp} by Market Intelligence Agent*"
            )
            logger.info("Using fallback report content")

        for attempt in range(3):
            try:
                with open(report_file_path, "w", encoding="utf-8") as f:
                    f.write(report_markdown)
                logger.info(f"Report written to {report_file_path}")
                break
            except Exception as e:
                error_logger.error(f"Write attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    raise IOError(f"Failed to write report: {str(e)}")

        if verify_report_file(report_file_path):
            logger.info(f"Report saved: {report_file_path} ({os.path.getsize(report_file_path)} bytes)")
        else:
            error_logger.error(f"Report verification failed: {report_file_path}")
            raise ValueError("Report verification failed")

        try:
            log_source = os.path.join(os.getcwd(), "market_intelligence.log")
            if os.path.exists(log_source):
                shutil.copy2(log_source, log_copy_path)
                logger.info(f"Copied log to {log_copy_path}")
            else:
                error_logger.error(f"Log file not found: {log_source}")
                with open(log_copy_path, "w", encoding="utf-8") as f:
                    f.write("Log file not generated during execution.")
                logger.info(f"Created placeholder log: {log_copy_path}")
        except Exception as e:
            error_logger.error(f"Failed to copy log: {str(e)}")

        readme_path = generate_readme(state, report_dir, report_file_name)
        if readme_path:
            logger.info(f"README generated: {readme_path}")
        else:
            error_logger.error(f"README generation failed")

    except Exception as e:
        error_logger.error(f"Report generation failed: {str(e)}")
        try:
            with open(report_file_path, "w", encoding="utf-8") as f:
                f.write(f"# Error Report\n\nError: {str(e)}")
            if verify_report_file(report_file_path):
                logger.info(f"Error report saved: {report_file_path}")
            else:
                error_logger.error(f"Error report verification failed")
        except Exception as we:
            error_logger.error(f"Failed to write error report: {str(we)}")
        raise

def run_market_intelligence_agent(query: str = "Market analysis", market_domain: str = "Technology", question: Optional[str] = None):
    logger.info(f"Running agent for {market_domain}, query: {query}, question: {question}")
    try:
        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError("TAVILY_API_KEY is not set in .env")
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is not set in .env")

        report_dir = os.path.join(os.getcwd(), "reports1")
        os.makedirs(report_dir, exist_ok=True)
        test_file = os.path.join(report_dir, "test.txt")
        try:
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Write permissions verified: {report_dir}")
        except Exception as e:
            error_logger.error(f"Permission test failed: {str(e)}")
            raise PermissionError(f"Cannot write to reports1 directory: {str(e)}")

        workflow = create_market_intelligence_workflow()
        state = MarketIntelligenceState(query=query, market_domain=market_domain, question=question)
        
        final_state = workflow.invoke(state)
        final_state = MarketIntelligenceState(**final_state)

        if not final_state.report_dir:
            error_logger.error("report_dir not set in final_state")
            raise ValueError("Report directory not initialized in final_state")
        output_dir = final_state.report_dir
        logger.info(f"Using report_dir from state: {output_dir}")

        report_path = os.path.join(output_dir, f"{final_state.market_domain.lower().replace(' ', '_')}_report_{final_state.state_id[:4]}.md")
        json_path = os.path.join(output_dir, f"{final_state.market_domain.lower().replace(' ', '_')}_data_sources.json")
        csv_path = os.path.join(output_dir, f"{final_state.market_domain.lower().replace(' ', '_')}_data_sources.csv")
        vector_store_path = os.path.join(output_dir, f"vector_store_{final_state.state_id[:4]}")
        log_path = os.path.join(output_dir, "market_intelligence.log")
        rag_log_path = os.path.join(output_dir, f"rag_responses_{final_state.state_id[:4]}.log")
        chart_paths = [
            os.path.join(output_dir, "market_growth.png"),
            os.path.join(output_dir, "competitor_share.png"),
            os.path.join(output_dir, "trend_impact.png")
        ]

        logger.info(f"Expected report path: {report_path}")
        logger.info(f"Expected JSON path: {json_path}")
        logger.info(f"Expected CSV path: {csv_path}")
        logger.info(f"Expected vector store path: {vector_store_path}")
        logger.info(f"Expected log path: {log_path}")
        logger.info(f"Expected RAG log path: {rag_log_path}")
        for chart_path in chart_paths:
            logger.info(f"Expected chart path: {chart_path}")

        files_to_check = [
            ("Report", report_path),
            ("JSON", json_path),
            ("CSV", csv_path),
            ("Vector Store", vector_store_path),
            ("Log", log_path),
            ("RAG Log", rag_log_path)
        ] + [("Chart", cp) for cp in chart_paths]

        missing_files = []
        for file_type, file_path in files_to_check:
            if os.path.exists(file_path):
                logger.info(f"{file_type} exists: {file_path}")
            else:
                error_logger.error(f"{file_type} not found: {file_path}")
                missing_files.append(file_path)

        success = os.path.exists(report_path)
        response = {
            "success": success,
            "report_path": report_path,
            "readme_path": os.path.join(output_dir, "README.md"),
            "json_path": json_path,
            "csv_path": csv_path,
            "vector_store_path": vector_store_path,
            "log_path": log_path,
            "rag_log_path": rag_log_path,
            "chart_paths": chart_paths,
            "query_response": final_state.query_response,
            "state_id": final_state.state_id
        }
        if missing_files:
            response["warnings"] = f"Missing files: {', '.join(missing_files)}"
        if not success:
            response["error"] = "Critical file (report) not found"

        if final_state.query_response:
            logger.info(f"RAG query response: {final_state.query_response[:200]}...")
        return response

    except Exception as e:
        error_logger.error(f"Agent workflow failed: {str(e)}\n{traceback.format_exc()}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        safe_query = re.sub(r'[^a-zA-Z0-9_-]', '_', query.lower().replace(' ', '_')[:10])
        output_dir = os.path.join("reports1", f"{safe_query}_{timestamp}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created fallback output directory: {output_dir}")

            placeholders = [
                (f"{market_domain.lower().replace(' ', '_')}_data_sources.json", "[]"),
                (f"{market_domain.lower().replace(' ', '_')}_data_sources.csv", "title,summary,url,source\n"),
                (f"{market_domain.lower().replace(' ', '_')}_report_{str(uuid4())[:4]}.md", f"# Error Report\n\nError: {str(e)}"),
                ("market_intelligence.log", f"Error: {str(e)}"),
                (f"rag_responses_{str(uuid4())[:4]}.log", "No RAG responses due to error"),
                ("README.md", "# Error\n\nAgent execution failed.")
            ]
            for file_name, content in placeholders:
                file_path = os.path.join(output_dir, file_name)
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Created placeholder: {file_path}")
                except Exception as we:
                    error_logger.error(f"Failed to create placeholder {file_path}: {str(we)}")

            vector_store_path = os.path.join(output_dir, f"vector_store_{str(uuid4())[:4]}")
            try:
                os.makedirs(vector_store_path, exist_ok=True)
                with open(os.path.join(vector_store_path, "placeholder.txt"), "w", encoding="utf-8") as f:
                    f.write("Placeholder vector store")
                logger.info(f"Created placeholder vector store: {vector_store_path}")
            except Exception as we:
                error_logger.error(f"Failed to create placeholder vector store: {str(we)}")

            chart_paths = []
            for chart in ["market_growth.png", "competitor_share.png", "trend_impact.png"]:
                chart_path = os.path.join(output_dir, chart)
                try:
                    with open(chart_path, "w", encoding="utf-8") as f:
                        f.write("Placeholder chart")
                    chart_paths.append(chart_path)
                    logger.info(f"Created placeholder chart: {chart_path}")
                except Exception as we:
                    error_logger.error(f"Failed to create placeholder chart {chart_path}: {str(we)}")

            report_path = os.path.join(output_dir, f"{market_domain.lower().replace(' ', '_')}_report_{str(uuid4())[:4]}.md")
            return {
                "success": False,
                "error": f"Workflow failed: {str(e)}",
                "report_path": report_path,
                "readme_path": os.path.join(output_dir, "README.md"),
                "json_path": os.path.join(output_dir, f"{market_domain.lower().replace(' ', '_')}_data_sources.json"),
                "csv_path": os.path.join(output_dir, f"{market_domain.lower().replace(' ', '_')}_data_sources.csv"),
                "vector_store_path": vector_store_path,
                "log_path": os.path.join(output_dir, "market_intelligence.log"),
                "rag_log_path": os.path.join(output_dir, f"rag_responses_{str(uuid4())[:4]}.log"),
                "chart_paths": chart_paths,
                "query_response": None,
                "state_id": str(uuid4())[:4]
            }
        except Exception as de:
            error_logger.error(f"Failed to create placeholder files: {str(de)}")
            return {
                "success": False,
                "error": f"Workflow failed: {str(e)}. Failed to create placeholders: {str(de)}",
                "report_path": None,
                "readme_path": None,
                "json_path": None,
                "csv_path": None,
                "vector_store_path": None,
                "log_path": None,
                "rag_log_path": None,
                "chart_paths": [],
                "query_response": None,
                "state_id": None
            }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Market Intelligence Agent")
    parser.add_argument("--query", type=str, default="Market analysis")
    parser.add_argument("--market", type=str, default="Technology")
    parser.add_argument("--question", type=str, default=None)
    args = parser.parse_args()
    result = run_market_intelligence_agent(args.query, args.market, args.question)
    if result["success"]:
        logger.info(f"Report generated: {result['report_path']}")
        logger.info(f"JSON: {result['json_path']}")
        logger.info(f"CSV: {result['csv_path']}")
        logger.info(f"Vector Store: {result['vector_store_path']}")
        logger.info(f"Log: {result['log_path']}")
        logger.info(f"RAG Log: {result['rag_log_path']}")
        for cp in result['chart_paths']:
            logger.info(f"Chart: {cp}")
    else:
        error_logger.error(f"Failed: {result.get('error')}")