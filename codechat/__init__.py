"""snowcode - Local RAG-powered codebase Q&A engine."""

__version__ = "0.3.1"

# Export agent_v2 for enhanced agent functionality
from .agent_v2 import (
    CodeAgent as CodeAgentV2,
    MultiAgentCoordinator,
    create_agent,
    create_coordinator,
    ToolRegistry,
    BaseTool,
    ToolResult,
    ToolPermission,
)
