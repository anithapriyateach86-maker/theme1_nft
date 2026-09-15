"""
LangChain tools for NFT (Non-Functional Requirements) review.
Defines two tools for analyzing requirements gaps and architecture risks.
"""

import os
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from data_loader import load_inputs


# Load environment variables
load_dotenv()


def get_llm():
    """
    Initialize and return a ChatAnthropic LLM instance.

    Reads CLAUDE_HAIKU_API_KEY and CLAUDE_BASE_URL from environment.
    Uses Claude Haiku 4.5 model with conservative temperature for focused analysis.

    Returns:
        ChatAnthropic: Configured LLM instance
    """
    api_key = os.getenv("CLAUDE_HAIKU_API_KEY")
    base_url = os.getenv("CLAUDE_BASE_URL")

    llm = ChatAnthropic(
        api_key=api_key,
        model_name="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        temperature=0.2,
        max_tokens=12000,
        anthropic_api_url=base_url,
    )
    return llm


@tool
def nfr_gap_checker(focus_area: str) -> str:
    """
    Identify missing, unclear, or underspecified Non-Functional Requirements.

    This tool analyzes user stories and application overview to detect gaps or
    ambiguities in Non-Functional Requirements (NFRs) such as:
    - Performance (response times, throughput)
    - Scalability (concurrent users, load handling)
    - Availability (uptime targets, SLAs)
    - Accessibility (keyboard navigation, screen readers)
    - Resiliency (fault tolerance, recovery)

    Use this tool when you need to assess whether NFRs are clearly defined
    in the requirements and business context.

    Args:
        focus_area: A short note on what aspects to focus on
                   (e.g., "payment processing", "peak load handling")

    Returns:
        str: Structured findings with Category, Risk/Gap, Evidence, Potential Impact,
             and Recommendation/Clarification for each finding
    """
    # Load input data
    data = load_inputs()
    user_stories = data["user_stories"]
    application_overview = data["application_overview"]

    # Build the prompt
    prompt = f"""You are an expert in Non-Functional Requirements (NFR) analysis.

Review the following user stories and application overview to identify missing, unclear,
or underspecified Non-Functional Requirements. Focus on: {focus_area}

OUTPUT FORMAT:
For each finding, provide exactly this structure:
- Category: [Performance | Scalability | Availability | Accessibility | Resiliency | Observability | Security | Other]
- Risk/Gap: [Brief description of what's missing or unclear]
- Evidence: [Quote or reference from the provided text, or state "Not mentioned"]
- Potential Impact: [How this gap could affect the business]
- Recommendation/Clarification: [What should be clarified or specified]

---

USER STORIES:
{user_stories}

---

APPLICATION OVERVIEW:
{application_overview}

---

Analyze thoroughly and be specific. Ground all findings only in the supplied text.
If something is not mentioned, say so explicitly rather than inferring.

IMPORTANT: Be concise. Avoid repeating the same point. Each finding should be distinct.
Do not add verbose preambles or unnecessary elaboration. Get to the findings directly."""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


@tool
def architecture_risk_scanner(focus_area: str) -> str:
    """
    Identify performance and resiliency risks in the proposed architecture.

    This tool analyzes the architecture and business volume expectations to detect:
    - Bottlenecks and single points of failure
    - Payment and retry handling gaps
    - Timeout strategy concerns
    - Peak-load readiness issues
    - External dependency risks

    Use this tool when you need to assess architectural readiness for the expected
    business volumes, especially during festival sales and traffic spikes.

    Args:
        focus_area: A short note on what aspects to focus on
                   (e.g., "payment processing", "inventory consistency")

    Returns:
        str: Structured findings with Category, Risk/Gap, Evidence, Potential Impact,
             and Recommendation/Clarification for each finding
    """
    # Load input data
    data = load_inputs()
    architecture = data["architecture"]
    business_volumes = data["business_volumes"]

    # Build the prompt
    prompt = f"""You are an expert in enterprise architecture and resilience engineering.

Review the following architecture design and business volume expectations to identify
performance and resiliency risks. Focus on: {focus_area}

Key concerns to evaluate:
- Bottlenecks for expected concurrent users and transaction rates
- Single points of failure (SPOF) and fallback strategies
- Payment processing reliability and idempotency
- Retry and timeout handling
- External dependency resilience (payment providers, logistics, notifications)
- Inventory consistency under high load
- Cache invalidation and consistency

OUTPUT FORMAT:
For each finding, provide exactly this structure:
- Category: [Performance | Scalability | Availability | Resiliency | Payment Handling | Observability | Other]
- Risk/Gap: [Brief description of the risk]
- Evidence: [Reference from architecture or business volumes, or state "Not mentioned"]
- Potential Impact: [How this risk could affect business operations]
- Recommendation/Clarification: [What mitigations or clarifications are needed]

---

ARCHITECTURE:
{architecture}

---

BUSINESS VOLUMES:
{business_volumes}

---

Analyze thoroughly and be specific. Ground all findings only in the supplied text.
If a detail is not mentioned, say so explicitly rather than assuming.

IMPORTANT: Be concise. Avoid repeating similar risks under different names. Each finding
should be distinct and unique. Do not add verbose preambles or unnecessary elaboration.
Focus on novel, substantive risks only."""

    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Test the tools
    print("Testing NFR Gap Checker...")
    result1 = nfr_gap_checker("payment processing and checkout")
    print(result1)
    print("\n" + "="*80 + "\n")
    print("Testing Architecture Risk Scanner...")
    result2 = architecture_risk_scanner("inventory and payment handling")
    print(result2)
