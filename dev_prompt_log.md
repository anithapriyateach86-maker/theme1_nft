# Development Prompt Log — ShopSphere NFT Reviewer Agent
 
This log documents the key prompts used with Claude Code (VS Code) to build the
agentic NFT Reviewer for ShopSphere.
 
## 1. Initial Build Prompt
Asked Claude Code to build a LangChain-based agent with:
- `data_loader.py` to read the 4 ShopSphere input files
- `tools.py` with two @tool-decorated functions: `nfr_gap_checker` and
  `architecture_risk_scanner`, each calling a shared `get_llm()` (ChatAnthropic,
  Claude Haiku 4.5 via Tekstac's gateway)
- `agent.py` orchestrating the agent, writing a final report and a tool trace
 
## 1b. Environment constraint discovered: create_tool_calling_agent unavailable
 
While building the initial agent.py, the first attempt used create_tool_calling_agent plus AgentExecutor, the older LangChain agent pattern. This failed with: ImportError: cannot import name create_tool_calling_agent from langchain.agents. This function has been removed in the installed LangChain version 1.4.0, which restructured its agent API around create_agent (LangGraph-based) instead. create_agent was used going forward as the correct, current API for this environment, not as a workaround.
 
## 2. Fix: Hard-coded tool execution
**Issue found:** The generated `agent.py` created an agent object but never invoked
it — instead, Python code called `nfr_gap_checker.invoke(...)` and
`architecture_risk_scanner.invoke(...)` directly in a fixed order, so tool
selection wasn't actually agent-driven.
**Prompt:** Asked Claude Code to fix `agent.py` so execution happens via
`agent.invoke(...)`, letting the LLM decide (based on tool descriptions and a
system prompt) which tool(s) to call, then extract the real tool-call trace from
the returned messages instead of hard-coding it.
 
## 3. Fix: Token limit for final synthesis
**Issue found:** Shared `max_tokens=4000` risked truncating the final synthesized
report (which combines both tools' findings into one large document).
**Prompt:** Asked Claude Code to raise `max_tokens` to 12000 in `get_llm()` and add
conciseness instructions to the prompts so the larger budget wasn't wasted on
repetition.
 
## 4. Fix: Identical tool trace outputs
**Issue found:** The tool trace showed identical preview text for both
`nfr_gap_checker` and `architecture_risk_scanner`, suggesting a bug in matching
each tool call to its correct result message.
**Prompt:** Asked Claude Code to add debug prints to inspect the raw message
chain, and fix `extract_tool_calls_from_messages` to correctly match each tool
call to its own `ToolMessage` by `tool_call_id`, with a warning instead of a
silent fallback if matching fails.
**Result:** Confirmed via debug output that the two tools now produce genuinely
distinct findings (different categories, evidence, and framing).
 
## 5. Fix: Report cleanup and count consistency
**Issue found:** The saved report had a leaked conversational preamble before the
title, and the Executive Summary's stated finding count didn't match the Summary
table's Total row.
**Prompt:** Asked Claude Code to (a) strip any text before the first `#` heading
before writing the report file, and (b) add an explicit counting instruction to
the synthesis prompt plus a Python-side validation check that warns if the
Executive Summary count and Summary table Total don't match.
**Result:** Verified both numbers now match (39) with no mismatch warning.
 
## 6. Cleanup: Remove debug prints
**Prompt:** Asked Claude Code to remove the temporary `[DEBUG]` print blocks used
to diagnose the tool-trace bug, while keeping the legitimate warning check and all
normal status prints (tool selection summary, output file paths, completion
message).
 
## Final Verified State
- Agent genuinely decides which tool(s) to call based on the review objective
  (not hard-coded).
- Tool trace shows distinct, correctly-matched outputs per tool.
- Final report (`output/nft_review_report.md`) is clean, evidence-grounded, and
  internally consistent (finding counts match).
- Tool trace (`output/agent_tool_trace.md`) documents tool name, input, and output
  per call, in order.