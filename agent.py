"""
Main NFT (Non-Functional Requirements) Review Agent for ShopSphere.

This script:
1. Loads environment variables
2. Initializes a LangChain tool-calling agent with NFT review tools
3. Invokes the agent to decide which tools are relevant for the review
4. Extracts tool calls, their inputs, and outputs from the agent's execution
5. Validates consistency between Executive Summary and Summary table counts
6. Outputs structured findings and tool traces
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
# NOTE: create_agent (from LangGraph) is the current API in LangChain 1.4.0+.
# The older create_tool_calling_agent + AgentExecutor pattern was removed in LangChain 1.x
# when the agent API was restructured around LangGraph-based agents. create_agent is
# used here intentionally as the correct, current API—not as a fallback or workaround.
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from tools import nfr_gap_checker, architecture_risk_scanner, get_llm


# Load environment variables
load_dotenv()


def create_output_directory():
    """Ensure the output directory exists."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def clean_report_content(report: str) -> str:
    """
    Strip leading conversational text before the first markdown heading.

    Ensures the report file starts directly with "# ShopSphere..." heading.

    Args:
        report: Raw report content possibly with leading text

    Returns:
        str: Report starting with the first "# " heading
    """
    # Find the first markdown heading (starts with "# ")
    lines = report.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('# '):
            # Return from this line onwards, preserving the heading
            return '\n'.join(lines[i:]).strip()

    # If no heading found, return as-is
    return report


def validate_finding_counts(report: str) -> dict:
    """
    Validate consistency between Executive Summary count and Summary table count.

    Searches for "X distinct findings" in Executive Summary and extracts the count
    from the Summary table's "Total Findings: Y" line, then compares them.

    Args:
        report: The final report content

    Returns:
        dict: {
            'executive_count': int or None,
            'summary_count': int or None,
            'header_count': int (actual count of finding headers),
            'matches': bool,
            'warning': str or None
        }
    """
    result = {
        'executive_count': None,
        'summary_count': None,
        'header_count': 0,
        'matches': False,
        'warning': None,
    }

    # Extract Executive Summary count (pattern: "X distinct findings")
    exec_match = re.search(r'(\d+)\s+distinct\s+findings', report, re.IGNORECASE)
    if exec_match:
        result['executive_count'] = int(exec_match.group(1))

    # Extract Summary table count (pattern: "TOTAL FINDINGS | **X**" or variations)
    summary_match = re.search(r'\*\*TOTAL\s+FINDINGS\*\*\s*\|\s*\*\*(\d+)\*\*', report, re.IGNORECASE)
    if not summary_match:
        summary_match = re.search(r'Total Findings:\s*(\d+)', report, re.IGNORECASE)
    if summary_match:
        result['summary_count'] = int(summary_match.group(1))

    # Count actual finding headers (numbered list items like "#### 1.", "### 1.", etc.)
    # This matches headers with finding numbers like "#### 1. Risk Title" or "### Finding 1"
    header_pattern = r'^#+\s+\d+[\.\s]'
    headers = re.findall(header_pattern, report, re.MULTILINE)
    result['header_count'] = len(headers)

    # Check consistency
    if result['executive_count'] is not None and result['summary_count'] is not None:
        if result['executive_count'] == result['summary_count']:
            result['matches'] = True
        else:
            result['warning'] = (
                f"MISMATCH: Executive Summary states {result['executive_count']} findings, "
                f"but Summary table shows {result['summary_count']} findings"
            )
    elif result['executive_count'] is None:
        result['warning'] = "Could not extract Executive Summary count"
    elif result['summary_count'] is None:
        result['warning'] = "Could not extract Summary table count"

    return result


def extract_tool_calls_from_messages(messages):
    """
    Extract tool calls and their results from the agent's message chain.

    Args:
        messages: List of message objects from the agent response

    Returns:
        list: List of dicts containing tool name, input, and output
    """
    tool_calls_log = []
    used_tool_message_indices = set()  # Track which ToolMessages we've already used

    i = 0
    while i < len(messages):
        msg = messages[i]

        # Look for AI messages that contain tool calls
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            # Find the corresponding tool result in subsequent messages
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', tool_call.get('tool', 'unknown'))
                tool_input = tool_call.get('args', {})
                tool_call_id = tool_call.get('id')

                tool_result = None
                found_match = False
                matched_index = None

                # Try to match by tool_call_id (STRICT matching)
                if tool_call_id:
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], ToolMessage):
                            msg_tool_call_id = getattr(messages[j], 'tool_call_id', None)
                            # Only match if IDs are present and equal
                            if msg_tool_call_id and msg_tool_call_id == tool_call_id:
                                # Make sure we haven't already used this ToolMessage
                                if j not in used_tool_message_indices:
                                    tool_result = messages[j].content
                                    found_match = True
                                    matched_index = j
                                    used_tool_message_indices.add(j)
                                    break

                # If no match found by ID, raise a visible warning
                if not found_match:
                    preview = str(tool_input).replace('\n', ' ')[:100]
                    print(f"⚠️  WARNING: Could not match ToolMessage for tool '{tool_name}'")
                    print(f"   tool_call_id: {tool_call_id}")
                    print(f"   input preview: {preview}")
                    tool_result = "[UNMATCHED - No corresponding ToolMessage found]"

                tool_calls_log.append({
                    "tool": tool_name,
                    "input": tool_input,
                    "output": tool_result or "",
                })

        i += 1

    return tool_calls_log


def format_tool_trace(tool_calls_log):
    """
    Format tool calls into a readable trace.

    Args:
        tool_calls_log: List of dicts with tool call info

    Returns:
        str: Formatted trace showing each tool call, input, and preview of output
    """
    trace = "# Agent Tool Trace\n\n"

    if not tool_calls_log:
        trace += "No tool calls made.\n"
        return trace

    for idx, call_info in enumerate(tool_calls_log, 1):
        trace += f"## Tool Call {idx}\n\n"
        trace += f"**Tool Name:** {call_info.get('tool', 'N/A')}\n\n"

        # Format the input nicely
        input_data = call_info.get('input', {})
        input_str = json.dumps(input_data, indent=2) if isinstance(input_data, dict) else str(input_data)
        trace += f"**Input:**\n```json\n{input_str}\n```\n\n"

        # Provide a preview of the output (first 2000 chars)
        output = call_info.get('output', '')
        if isinstance(output, str):
            preview = output[:2000] + "..." if len(output) > 2000 else output
        else:
            preview = str(output)[:2000]

        trace += f"**Output Preview:**\n```\n{preview}\n```\n\n"

    trace += "---\n\n"
    trace += "## Summary\n\n"
    trace += (
        f"The agent made **{len(tool_calls_log)}** tool call(s) to complete the NFT review:\n\n"
    )

    for idx, call_info in enumerate(tool_calls_log, 1):
        tool_name = call_info.get('tool', 'unknown')
        input_data = call_info.get('input', {})
        focus = input_data.get('focus_area', 'no focus specified')
        trace += f"{idx}. **{tool_name}** — focus: {focus}\n"

    trace += "\nThese findings were synthesized by the agent into a single comprehensive report.\n"

    return trace


def run_nft_review():
    """
    Execute the NFT review agent.

    Invokes the agent to decide which tools are relevant and in what order,
    then extracts tool calls and synthesizes findings.

    Returns:
        tuple: (final_report, tool_calls_log, chosen_tools)
    """

    # Define the system prompt for the agent
    # This prompt describes both tools and asks the LLM to decide which are relevant
    system_prompt = """You are an expert NFT (Non-Functional Requirements) Reviewer for ShopSphere,
an e-commerce platform launching in multiple regions with expected extreme traffic spikes during
festival sales.

You have two specialized tools available:

1. **nfr_gap_checker** - Use this tool to identify missing, unclear, or underspecified
   Non-Functional Requirements by analyzing user stories and application overview.
   This tool detects gaps in performance, scalability, availability, accessibility,
   resiliency, and observability specifications.

   When to use: Call this tool when you need to assess whether the requirements are
   clearly defined and whether key NFRs are specified for critical journeys like
   payment processing, inventory management, and user experience during peak load.

2. **architecture_risk_scanner** - Use this tool to identify performance and resiliency
   risks in the proposed architecture by analyzing the design and business volume expectations.
   This tool detects bottlenecks, single points of failure, payment handling gaps,
   timeout strategy issues, and external dependency vulnerabilities.

   When to use: Call this tool when you need to assess whether the architecture can
   handle the expected load (75,000 concurrent users, 95,000 orders/hour during festival),
   maintain consistency, and survive failures.

YOUR TASK:
You are performing a comprehensive Non-Functional Requirements review for ShopSphere.
Based on the full ShopSphere input package (application overview, user stories,
architecture, and business volumes), DECIDE which tool(s) are relevant and CALL THEM.

You may decide to call:
- Only the nfr_gap_checker
- Only the architecture_risk_scanner
- BOTH tools (recommended for comprehensive coverage)
- Neither if the inputs already fully address all NFRs

After gathering findings from the tool(s) you call, synthesize them into a final report.

REPORT STRUCTURE:
1. Title: "# ShopSphere Non-Functional Requirements Review"
2. Executive Summary (1-2 paragraphs):
   - State the total number of distinct findings identified (COUNT THEM ACCURATELY)
   - Highlight the most critical findings
3. Group findings by Category — USE ONLY THESE EXACT NAMES:
   Performance, Scalability, Availability, Accessibility, Resiliency, Observability,
   Payment Handling, Security, Other
   DO NOT create variant names like "Resiliency Gaps" or "Resiliency Specification".
   If findings appear similar across categories, use the one best-fit category.
4. For each finding, include:
   - **Risk/Gap:** [description]
   - **Evidence:** [quote or reference from provided artifacts]
   - **Potential Impact:** [business impact]
   - **Recommendation:** [clarification or mitigation]
5. MANDATORY DEDUPLICATION AND MERGING: Eliminate duplicate and near-duplicate findings.
   When multiple findings describe the same underlying issue (e.g., two findings about
   payment idempotency, duplicate transactions, order state consistency, or database failover),
   merge them into ONE finding that combines all Evidence, Impact, and Recommendation points.
   Do not create separate entries for the same issue even if they use different wording.
   For example: merge "Payment Idempotency" + "Duplicate Transaction Prevention" into
   a single "Payment Processing Idempotency" finding with consolidated details.
   This is MANDATORY—check each finding against others to eliminate redundancy.
6. End with "## Summary" section that lists all findings with their categories
   (this enables easy counting and verification)
7. End with "## Observability Recommendations" section

IMPORTANT STYLE GUIDANCE:
- Be thorough and specific, grounding all findings in the supplied artifacts.
- Be CONCISE: avoid repetition, verbose preambles, or redundant points.
- MERGE findings describing the same underlying issue. Do not list "Payment Idempotency"
  and "Duplicate Transaction Prevention" as separate findings if they describe the same gap.
  Instead, merge them into one finding with combined Evidence, Impact, and Recommendations.
- Eliminate duplicates and consolidate similar findings across tool outputs.
- Each finding should be distinct and substantive. Do not pad the report.
- Use your full token budget (12000) for depth and specificity, not verbosity.
- Get to the findings directly without lengthy introductions.

CRITICAL: ELIMINATE DUPLICATES AND COUNT FINDINGS ACCURATELY
- BEFORE counting, systematically review all findings and merge any that describe the same
  underlying issue. Examples: if two findings discuss payment idempotency, merge them;
  if two discuss database failover, merge them. Do this BEFORE finalizing the report.
- Count the total number of MERGED findings (not the original list from tools).
- State this number in the Executive Summary (e.g., "This review identifies X findings")
- Ensure the Summary section table lists ALL and ONLY these deduplicated findings
- After writing all findings and the Summary table, count the exact number of individual
  findings you wrote (each numbered entry across all categories), and use that exact same
  number in both the Executive Summary and the Total row of the Summary table
- Do not estimate — count them precisely from what you actually wrote
- The Executive Summary count MUST match the Summary table count exactly"""

    # Initialize the LLM
    llm = get_llm()

    # Create the agent using LangGraph-based create_agent API (LangChain 1.4.0+).
    # This replaces the deprecated create_tool_calling_agent + AgentExecutor pattern
    # which was removed when LangChain 1.x restructured its agents around LangGraph.
    # create_agent is the current standard for tool-calling agents in this version.
    tools = [nfr_gap_checker, architecture_risk_scanner]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # Define the user input for the agent
    # This is open-ended to let the agent decide which tools to call
    user_input = (
        "Perform a comprehensive Non-Functional Requirements review for ShopSphere. "
        "Based on the review objective and available tools, decide which tool(s) are most relevant. "
        "Call the necessary tool(s) to gather findings, then synthesize them into a single "
        "structured Markdown report organized by category."
    )

    # Invoke the agent
    print("Invoking agent with decision authority...")
    print(f"  Input: {user_input[:80]}...\n")

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Extract messages from the result
    messages = result.get("messages", [])

    # Extract tool calls from the message chain
    tool_calls_log = extract_tool_calls_from_messages(messages)

    # Determine which tools were actually called
    chosen_tools = list(set(call['tool'] for call in tool_calls_log))

    # Extract the final response (last AI message content)
    final_report = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_report = msg.content
            break

    return final_report, tool_calls_log, chosen_tools


def main():
    """Main execution function."""
    print("="*80)
    print("ShopSphere NFT (Non-Functional Requirements) Review Agent")
    print("="*80)
    print()

    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    print()

    # Run the review
    print("Running NFT review with agent-driven tool selection...\n")
    final_report, tool_calls_log, chosen_tools = run_nft_review()

    # Show which tools the agent chose
    print()
    print("="*80)
    print("Agent Tool Selection Summary")
    print("="*80)
    if chosen_tools:
        print(f"✓ Agent chose to call {len(chosen_tools)} tool(s):")
        for tool in chosen_tools:
            print(f"  - {tool}")
    else:
        print("✗ Agent made no tool calls (check system prompt and inputs)")
    print()

    # Show number of findings
    if tool_calls_log:
        print(f"✓ Tool calls made: {len(tool_calls_log)}")
        for idx, call in enumerate(tool_calls_log, 1):
            focus = call.get('input', {}).get('focus_area', 'N/A')
            print(f"  {idx}. {call.get('tool', 'unknown')} (focus: {focus[:50]}...)")
    else:
        print("! No tool calls were captured")
    print()

    # Write the final report (strip leading conversational text)
    cleaned_report = clean_report_content(final_report)
    report_path = output_dir / "nft_review_report.md"
    with open(report_path, "w") as f:
        f.write(cleaned_report)
    print(f"✓ Final report written to: {report_path}")

    # Validate finding counts for consistency
    print()
    validation = validate_finding_counts(cleaned_report)
    print("Finding Count Validation:")
    print(f"  Executive Summary count: {validation['executive_count']}")
    print(f"  Summary table count: {validation['summary_count']}")
    print(f"  Finding headers found: {validation['header_count']}")
    if validation['warning']:
        print(f"  ⚠️  {validation['warning']}")
    else:
        print(f"  ✓ Counts are consistent")
    print()

    # Write the tool trace
    trace_path = output_dir / "agent_tool_trace.md"
    trace_content = format_tool_trace(tool_calls_log)
    with open(trace_path, "w") as f:
        f.write(trace_content)
    print(f"✓ Tool trace written to:   {trace_path}")
    print()

    # Print confirmation
    print("="*80)
    print("✓ Review Complete!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
