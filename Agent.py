import getpass
import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

class MarketIntelligenceState(BaseModel):
    raw_news_data: List[Dict[str, Any]] = Field(default_factory=list)
    competitor_data: List[Dict[str, Any]] = Field(default_factory=list)
    market_trends: List[Dict[str, Any]] = Field(default_factory=list)
    opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    strategic_recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    query: Optional[str] = None
    query_response: Optional[str] = None

def market_data_collector(state: MarketIntelligenceState) -> Dict[str, Any]:
    news_urls = [
        "https://www.edsurge.com/",
        "https://edtechmagazine.com/"
    ]

    raw_news_data = []
    for url in news_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                raw_news_data.append({
                    "source": url,
                    "title": doc.metadata.get("title", "No title"),
                    "date": doc.metadata.get("publish_date", "Unknown"),
                    "summary": doc.page_content[:300],
                    "url": url
                })
        except Exception as e:
            raw_news_data.append({
                "source": url,
                "title": "Failed to load",
                "date": "",
                "summary": str(e),
                "url": url
            })

    simulated_competitor_data = [
        {
            "name": "LearnSmart",
            "product": "SmartLearn K-12",
            "focus_areas": ["Personalized learning", "Math", "Science"],
            "pricing_model": "Subscription: $9.99/student/month",
            "recent_updates": "Added AR science experiments feature",
            "user_sentiment": "4.5/5",
            "market_share_estimate": "12%"
        }
    ]

    return {
        "raw_news_data": raw_news_data,
        "competitor_data": simulated_competitor_data
    }

def trend_analyzer(state: MarketIntelligenceState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze K-12 EdTech data. Return JSON array of trends: trend_name, description, supporting_evidence, estimated_impact, timeframe."),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    trends = chain.invoke({"input": json.dumps({
        "news_data": state.raw_news_data,
        "competitor_data": state.competitor_data
    })})
    return {"market_trends": trends}

def opportunity_identifier(state: MarketIntelligenceState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Identify market opportunities from trends, competitors, and news. Return JSON array: opportunity_name, description, target_segment, competitive_advantage, estimated_potential, timeframe_to_capture."),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    opportunities = chain.invoke({"input": json.dumps({
        "market_trends": state.market_trends,
        "competitor_data": state.competitor_data,
        "news_data": state.raw_news_data
    })})
    return {"opportunities": opportunities}

def strategy_recommender(state: MarketIntelligenceState) -> Dict[str, Any]:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Recommend strategies for each opportunity. Return JSON array: strategy_title, description, implementation_steps, expected_outcome, resource_requirements, priority_level, success_metrics."),
        ("human", "{input}")
    ])
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    strategies = chain.invoke({"input": json.dumps({
        "opportunities": state.opportunities,
        "market_trends": state.market_trends,
        "competitor_data": state.competitor_data
    })})
    return {"strategic_recommendations": strategies}

def query_engine(state: MarketIntelligenceState) -> Dict[str, Any]:
    if not state.query:
        return {}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an EdTech analyst. Answer the query using market data. If insufficient, say so."),
        ("human", "Query: {query}\n\nData: {data}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "query": state.query,
        "data": json.dumps({
            "query": state.query,
            "strategic_recommendations": state.strategic_recommendations,
            "opportunities": state.opportunities,
            "market_trends": state.market_trends,
            "competitor_data": state.competitor_data,
            "news_data": state.raw_news_data
        })
    })
    return {"query_response": answer}

def create_market_intelligence_workflow():
    workflow = StateGraph(MarketIntelligenceState)
    workflow.add_node("market_data_collector", market_data_collector)
    workflow.add_node("trend_analyzer", trend_analyzer)
    workflow.add_node("opportunity_identifier", opportunity_identifier)
    workflow.add_node("strategy_recommender", strategy_recommender)
    workflow.add_node("query_engine", query_engine)

    workflow.set_entry_point("market_data_collector")
    workflow.add_edge("market_data_collector", "trend_analyzer")
    workflow.add_edge("trend_analyzer", "opportunity_identifier")
    workflow.add_edge("opportunity_identifier", "strategy_recommender")
    workflow.add_edge("strategy_recommender", "query_engine")
    workflow.add_edge("query_engine", END)

    return workflow.compile()

def generate_market_intelligence_report(state: MarketIntelligenceState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a well-structured markdown market report from the following data."),
        ("human", "Data: {data}")
    ])
    chain = prompt | llm | StrOutputParser()
    report_markdown = chain.invoke({
        "data": json.dumps({
            "market_trends": state.market_trends,
            "opportunities": state.opportunities,
            "strategic_recommendations": state.strategic_recommendations,
            "competitor_data": state.competitor_data,
            "news_data": state.raw_news_data
        })
    })
    with open("edtech_market_intelligence_report.md", "w", encoding="utf-8") as f:
        f.write(report_markdown)
    print("✅ Full report saved to 'edtech_market_intelligence_report.md'")

def run_market_intelligence_agent(query: Optional[str] = "What are the most promising opportunities in middle school math education?"):
    print("Running EdTech Market Intelligence Agent...")
    workflow = create_market_intelligence_workflow()
    final_state = workflow.invoke(MarketIntelligenceState(query=query))
    final_state = MarketIntelligenceState(**final_state)
    print("Market intelligence analysis complete.")
    print(f"Query response: {final_state.query_response}")
    generate_market_intelligence_report(final_state)

if __name__ == "__main__":
    run_market_intelligence_agent()
