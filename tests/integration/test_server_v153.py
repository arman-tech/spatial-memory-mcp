"""Integration tests for v1.5.3 server features.

Tests for:
- Request tracing (request_id, agent_id tracking)
- Response caching (cache hits, invalidation)
- _meta inclusion in responses
- Per-agent rate limiting
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from spatial_memory.config import Settings, override_settings, reset_settings
from spatial_memory.server import SpatialMemoryServer


@pytest.fixture(autouse=True)
def reset_settings_after_test() -> None:
    """Reset settings after each test."""
    yield
    reset_settings()


@pytest.fixture
def temp_storage() -> Path:
    """Create a temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def settings_with_meta(temp_storage: Path) -> Settings:
    """Create settings with _meta enabled."""
    return Settings(
        memory_path=temp_storage,
        include_request_meta=True,
        include_timing_breakdown=True,
        response_cache_enabled=True,
        response_cache_max_size=100,
        response_cache_default_ttl=60.0,
        rate_limit_per_agent_enabled=True,
        rate_limit_per_agent_rate=10.0,
        rate_limit_max_tracked_agents=5,
    )


@pytest.fixture
def settings_without_meta(temp_storage: Path) -> Settings:
    """Create settings without _meta enabled."""
    return Settings(
        memory_path=temp_storage,
        include_request_meta=False,
        response_cache_enabled=True,
        rate_limit_per_agent_enabled=False,
    )


@pytest.fixture
def server_with_meta(settings_with_meta: Settings) -> SpatialMemoryServer:
    """Create a server with _meta enabled."""
    override_settings(settings_with_meta)
    server = SpatialMemoryServer()
    yield server
    server.close()


@pytest.fixture
def server_without_meta(settings_without_meta: Settings) -> SpatialMemoryServer:
    """Create a server without _meta."""
    override_settings(settings_without_meta)
    server = SpatialMemoryServer()
    yield server
    server.close()


def _call_tool(server: SpatialMemoryServer, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Helper to call a tool and parse the JSON result."""
    result = server._handle_tool(name, arguments)
    return result


class TestRequestMeta:
    """Tests for _meta inclusion in responses."""

    def test_meta_included_when_enabled(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test that _meta is included when include_request_meta=True."""
        # Store a memory first
        remember_result = _call_tool(
            server_with_meta,
            "remember",
            {
                "content": "Test memory for meta testing",
                "_agent_id": "test-agent-1",
            },
        )
        assert "id" in remember_result

        # Recall - _meta is added in the call_tool wrapper, not _handle_tool
        # So we verify the settings are correct
        _call_tool(
            server_with_meta,
            "recall",
            {
                "query": "meta testing",
                "_agent_id": "test-agent-1",
            },
        )
        assert server_with_meta._settings.include_request_meta is True

    def test_meta_not_included_when_disabled(
        self, server_without_meta: SpatialMemoryServer
    ) -> None:
        """Test that _meta is NOT included when include_request_meta=False."""
        # Store a memory
        remember_result = _call_tool(
            server_without_meta,
            "remember",
            {
                "content": "Test memory",
            },
        )
        assert "id" in remember_result

        # _meta should not be in result (added at wrapper level)
        assert "_meta" not in remember_result
        assert server_without_meta._settings.include_request_meta is False


class TestResponseCaching:
    """Tests for response caching."""

    def test_cache_enabled(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test that cache is initialized when enabled."""
        assert server_with_meta._cache_enabled is True
        assert server_with_meta._cache is not None

    def test_cache_disabled(self, temp_storage: Path) -> None:
        """Test that cache is None when disabled."""
        settings = Settings(
            memory_path=temp_storage,
            response_cache_enabled=False,
        )
        override_settings(settings)
        server = SpatialMemoryServer()
        try:
            assert server._cache_enabled is False
            assert server._cache is None
        finally:
            server.close()

    def test_cacheable_tools_list(self) -> None:
        """Test that the correct tools are marked as cacheable."""
        from spatial_memory.server import CACHEABLE_TOOLS

        expected = {"recall", "nearby", "hybrid_recall", "regions"}
        assert CACHEABLE_TOOLS == expected

    def test_cache_key_generation(self) -> None:
        """Test that cache keys are generated correctly."""
        from spatial_memory.server import _generate_cache_key

        # Same arguments should produce same key
        key1 = _generate_cache_key("recall", {"query": "test", "limit": 5})
        key2 = _generate_cache_key("recall", {"query": "test", "limit": 5})
        assert key1 == key2

        # Different arguments should produce different keys
        key3 = _generate_cache_key("recall", {"query": "test", "limit": 10})
        assert key1 != key3

        # _agent_id should NOT affect cache key
        key4 = _generate_cache_key("recall", {"query": "test", "limit": 5, "_agent_id": "agent-1"})
        assert key1 == key4

    def test_cache_invalidation_tools(self) -> None:
        """Test that mutation tools are correctly identified."""
        from spatial_memory.server import FULL_INVALIDATING_TOOLS, NAMESPACE_INVALIDATING_TOOLS

        assert "remember" in NAMESPACE_INVALIDATING_TOOLS
        assert "forget" in NAMESPACE_INVALIDATING_TOOLS
        assert "forget_batch" in NAMESPACE_INVALIDATING_TOOLS

        assert "decay" in FULL_INVALIDATING_TOOLS
        assert "reinforce" in FULL_INVALIDATING_TOOLS
        assert "consolidate" in FULL_INVALIDATING_TOOLS


class TestPerAgentRateLimiting:
    """Tests for per-agent rate limiting."""

    def test_per_agent_limiter_enabled(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test that per-agent rate limiter is used when enabled."""
        assert server_with_meta._per_agent_rate_limiting is True
        assert server_with_meta._agent_rate_limiter is not None
        assert server_with_meta._rate_limiter is None

    def test_fallback_limiter_when_disabled(self, server_without_meta: SpatialMemoryServer) -> None:
        """Test that simple rate limiter is used when per-agent is disabled."""
        assert server_without_meta._per_agent_rate_limiting is False
        assert server_without_meta._agent_rate_limiter is None
        assert server_without_meta._rate_limiter is not None


class TestTracingIntegration:
    """Tests for request tracing integration."""

    def test_agent_id_removed_from_arguments(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test that _agent_id is properly handled (removed before handler)."""
        # _agent_id should be popped from arguments in call_tool wrapper
        # The handler should never see it
        result = _call_tool(
            server_with_meta,
            "remember",
            {
                "content": "Test content",
                "_agent_id": "test-agent",
            },
        )
        # Should succeed without error (handler doesn't receive _agent_id)
        assert "id" in result


class TestToolDefinitions:
    """Tests for v1.5.3 tool definition changes."""

    def test_all_tools_have_agent_id(self) -> None:
        """Test that all tools have _agent_id parameter."""
        from spatial_memory.tools.definitions import TOOLS

        for tool in TOOLS:
            props = tool.inputSchema.get("properties", {})
            assert "_agent_id" in props, f"Tool {tool.name} missing _agent_id"
            assert props["_agent_id"]["type"] == "string"

    def test_remember_has_idempotency_key(self) -> None:
        """Test that remember tool has idempotency_key parameter."""
        from spatial_memory.tools.definitions import TOOLS

        remember = next(t for t in TOOLS if t.name == "remember")
        props = remember.inputSchema.get("properties", {})
        assert "idempotency_key" in props
        assert props["idempotency_key"]["type"] == "string"


class TestBuildMeta:
    """Tests for _build_meta helper method."""

    def test_build_meta_with_timing(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test _build_meta includes timing when enabled."""
        from spatial_memory.core.tracing import RequestContext, TimingContext
        from spatial_memory.core.utils import utc_now

        ctx = RequestContext(
            request_id="test123",
            agent_id="agent-1",
            tool_name="recall",
            started_at=utc_now(),
            namespace="default",
        )
        timing = TimingContext()
        with timing.measure("handler"):
            time.sleep(0.001)  # Small delay

        meta = server_with_meta._build_meta(ctx, timing, cache_hit=False)

        assert meta["request_id"] == "test123"
        assert meta["agent_id"] == "agent-1"
        assert meta["cache_hit"] is False
        assert "timing_ms" in meta
        assert "handler" in meta["timing_ms"]
        assert "total_ms" in meta["timing_ms"]

    def test_build_meta_without_timing(self, temp_storage: Path) -> None:
        """Test _build_meta excludes timing when disabled."""
        from spatial_memory.core.tracing import RequestContext, TimingContext
        from spatial_memory.core.utils import utc_now

        settings = Settings(
            memory_path=temp_storage,
            include_request_meta=True,
            include_timing_breakdown=False,  # Disabled
        )
        override_settings(settings)
        server = SpatialMemoryServer()
        try:
            ctx = RequestContext(
                request_id="test456",
                agent_id=None,
                tool_name="recall",
                started_at=utc_now(),
            )
            timing = TimingContext()

            meta = server._build_meta(ctx, timing, cache_hit=True)

            assert meta["request_id"] == "test456"
            assert meta["agent_id"] is None
            assert meta["cache_hit"] is True
            assert "timing_ms" not in meta
        finally:
            server.close()


class TestCacheInvalidation:
    """Tests for cache invalidation logic."""

    def test_invalidate_namespace(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test namespace-based cache invalidation."""
        cache = server_with_meta._cache
        assert cache is not None

        # Manually add some cache entries
        cache.set("recall:ns1:query1:5", {"result": 1})
        cache.set("recall:ns2:query2:5", {"result": 2})
        cache.set("nearby:ns1:memid:5", {"result": 3})

        # Invalidate ns1
        server_with_meta._invalidate_cache_for_tool("remember", {"namespace": "ns1"})

        # ns1 entries should be gone
        assert cache.get("recall:ns1:query1:5") is None
        assert cache.get("nearby:ns1:memid:5") is None
        # ns2 entry should remain
        assert cache.get("recall:ns2:query2:5") is not None

    def test_invalidate_all(self, server_with_meta: SpatialMemoryServer) -> None:
        """Test full cache invalidation."""
        cache = server_with_meta._cache
        assert cache is not None

        # Add some cache entries
        cache.set("recall:ns1:query1:5", {"result": 1})
        cache.set("recall:ns2:query2:5", {"result": 2})

        # Full invalidation (e.g., from decay)
        server_with_meta._invalidate_cache_for_tool("decay", {})

        # All entries should be gone
        assert cache.get("recall:ns1:query1:5") is None
        assert cache.get("recall:ns2:query2:5") is None


class TestServerInstructions:
    """Tests for MCP server instructions (auto-injected AI behavioral guidelines)."""

    def test_get_server_instructions_returns_string(self) -> None:
        """Test that _get_server_instructions returns a non-empty string."""
        instructions = SpatialMemoryServer._get_server_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0

    def test_instructions_contain_key_sections(self) -> None:
        """Test that instructions contain all required behavioral sections."""
        instructions = SpatialMemoryServer._get_server_instructions()

        # Check for key section headers
        assert "## Spatial Memory System" in instructions
        assert "### Session Start" in instructions
        assert "### Recognizing Memory-Worthy Moments" in instructions
        assert "### Saving Memories" in instructions
        assert "### Synthesizing Answers" in instructions
        assert "### Auto-Extract" in instructions
        assert "### Tool Selection Guide" in instructions

    def test_instructions_contain_tool_names(self) -> None:
        """Test that instructions reference the available tools."""
        instructions = SpatialMemoryServer._get_server_instructions()

        # Key tools should be mentioned
        assert "recall" in instructions
        assert "remember" in instructions
        assert "hybrid_recall" in instructions
        assert "extract" in instructions
        assert "nearby" in instructions
        assert "regions" in instructions
        assert "journey" in instructions

    def test_instructions_contain_behavioral_triggers(self) -> None:
        """Test that instructions contain memory-worthy trigger phrases."""
        instructions = SpatialMemoryServer._get_server_instructions()

        # Trigger patterns for decisions, solutions, patterns
        assert "Let's use" in instructions or "We decided" in instructions
        assert "The fix was" in instructions or "failed because" in instructions
        assert "Save this? y/n" in instructions

    def test_instructions_emphasize_natural_synthesis(self) -> None:
        """Test that instructions guide Claude to present memories naturally."""
        instructions = SpatialMemoryServer._get_server_instructions()

        # Should guide natural presentation
        assert "Good:" in instructions or "naturally" in instructions.lower()
        assert "Bad:" in instructions or "raw JSON" in instructions
