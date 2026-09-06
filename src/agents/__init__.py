"""Agent package exports kept lazy so local specialist models need no LLM config."""

DEFAULT_AGENT = "research-assistant"


def __getattr__(name):
    if name in {"get_agent", "load_agent", "get_all_agent_info"}:
        from agents.agents import get_agent, get_all_agent_info, load_agent
        return {"get_agent": get_agent, "load_agent": load_agent, "get_all_agent_info": get_all_agent_info}[name]
    raise AttributeError(name)


__all__ = ["get_agent", "load_agent", "get_all_agent_info", "DEFAULT_AGENT"]
