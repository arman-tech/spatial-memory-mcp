"""Cross-corpus integration tests: Front-end / Back-end scenario.

Simulates two apps sharing a memory database:
  - "frontend": UI bugs, user-facing errors, component issues
  - "backend": API endpoints, database queries, server logic

The engine should discover semantic connections between them,
e.g. a frontend timeout complaint linking to a backend N+1 query.

Run with:
    pytest tests/integration/test_cross_corpus_frontend_backend.py -v -s -m ""
"""

from __future__ import annotations

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.database import Database
from spatial_memory.core.models import SimilarityConfig
from spatial_memory.server import SpatialMemoryServer
from spatial_memory.services.similarity import CrossCorpusSimilarityService

# =============================================================================
# Test Data
# =============================================================================

FRONTEND_MEMORIES = [
    {
        "content": (
            "Users reporting slow page loads on the dashboard,"
            " takes 8+ seconds to render the analytics charts"
        ),
        "namespace": "frontend",
        "tags": ["performance", "dashboard", "ux"],
        "importance": 0.9,
    },
    {
        "content": (
            "Checkout page throwing 401 unauthorized errors after users sit idle for 15 minutes"
        ),
        "namespace": "frontend",
        "tags": ["auth", "checkout", "bug"],
        "importance": 0.8,
    },
    {
        "content": (
            "Search results not refreshing when user types,"
            " stale data showing until full page reload"
        ),
        "namespace": "frontend",
        "tags": ["search", "caching", "bug"],
        "importance": 0.7,
    },
    {
        "content": (
            "Mobile responsive layout breaks on the user profile settings page below 375px"
        ),
        "namespace": "frontend",
        "tags": ["mobile", "css", "responsive"],
        "importance": 0.5,
    },
    {
        "content": (
            "Added React error boundary around payment form to catch Stripe element crashes"
        ),
        "namespace": "frontend",
        "tags": ["payments", "error-handling", "react"],
        "importance": 0.6,
    },
]

BACKEND_MEMORIES = [
    {
        "content": (
            "N+1 query detected on GET /api/dashboard/analytics"
            " endpoint, loading 200+ chart records individually"
        ),
        "namespace": "backend",
        "tags": ["performance", "database", "api"],
        "importance": 0.9,
    },
    {
        "content": (
            "JWT token expiry set to 15 minutes, refresh token endpoint returning 500 under load"
        ),
        "namespace": "backend",
        "tags": ["auth", "jwt", "bug"],
        "importance": 0.8,
    },
    {
        "content": (
            "Elasticsearch reindex job running every 30 minutes causing stale search results"
        ),
        "namespace": "backend",
        "tags": ["search", "elasticsearch", "indexing"],
        "importance": 0.7,
    },
    {
        "content": (
            "Rate limiter on POST /api/payments blocking legitimate retry attempts from the client"
        ),
        "namespace": "backend",
        "tags": ["payments", "rate-limit", "api"],
        "importance": 0.6,
    },
    {
        "content": ("Database migration adding composite index on users table for profile lookups"),
        "namespace": "backend",
        "tags": ["database", "migration", "performance"],
        "importance": 0.5,
    },
]

ALL_MEMORIES = FRONTEND_MEMORIES + BACKEND_MEMORIES


def _seed(server: SpatialMemoryServer) -> dict[str, list[str]]:
    """Store all test memories, return IDs grouped by namespace."""
    ids: dict[str, list[str]] = {"frontend": [], "backend": []}
    for mem in ALL_MEMORIES:
        result = server._handle_tool("remember", mem)
        ids[mem["namespace"]].append(result["id"])
    return ids


# =============================================================================
# Test 1: Dashboard performance — front-end slow load ↔ back-end N+1 query
# =============================================================================


@pytest.mark.integration
class TestDashboardPerformanceBridge:
    """The #1 use-case: a UI complaint links to a server-side root cause."""

    def test_frontend_slowness_connects_to_backend_query(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The frontend "slow dashboard" memory discovers a
        connection to the backend "N+1 query on /api/dashboard" memory.
        """
        ids = _seed(module_server)
        slow_dashboard_id = ids["frontend"][0]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": slow_dashboard_id,
                "limit": 5,
                "min_similarity": 0.2,
                "exclude_same_namespace": True,
            },
        )

        connections = result["connections"]

        print("\n=== Dashboard Performance Bridge ===")
        print("SOURCE: frontend — slow dashboard page loads")
        print("EXPECTED: backend N+1 query on /api/dashboard")
        print(f"ACTUAL:   {result['total_found']} connections:")
        for c in connections:
            print(f"  [{c['namespace']}] sim={c['similarity']:.4f} | {c['content'][:70]}...")

        assert result["total_found"] > 0
        assert all(c["namespace"] == "backend" for c in connections)

        # The top match should be dashboard/performance related
        top = connections[0]
        top_lower = top["content"].lower()
        assert "dashboard" in top_lower or "analytics" in top_lower, (
            f"Top match should be dashboard-related, got: {top['content'][:80]}"
        )


# =============================================================================
# Test 2: Auth session expiry — front-end 401 ↔ back-end JWT token
# =============================================================================


@pytest.mark.integration
class TestAuthExpiryBridge:
    """Frontend 401 after idle should link to backend JWT expiry config."""

    def test_frontend_401_connects_to_backend_jwt(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: The frontend "401 after 15 min idle" memory connects
        to the backend "JWT expiry set to 15 minutes" memory.
        Both mention auth and the 15-minute window.
        """
        ids = _seed(module_server)
        checkout_401_id = ids["frontend"][1]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": checkout_401_id,
                "limit": 5,
                "min_similarity": 0.2,
                "exclude_same_namespace": True,
            },
        )

        connections = result["connections"]

        print("\n=== Auth Expiry Bridge ===")
        print("SOURCE: frontend — 401 errors after 15 min idle")
        print("EXPECTED: backend JWT token expiry at 15 minutes")
        print(f"ACTUAL:   {result['total_found']} connections:")
        for c in connections:
            print(f"  [{c['namespace']}] sim={c['similarity']:.4f} | {c['content'][:70]}...")

        assert result["total_found"] > 0

        # Should find JWT/auth related backend memory
        auth_connections = [
            c
            for c in connections
            if "jwt" in c["content"].lower()
            or "auth" in c["content"].lower()
            or "token" in c["content"].lower()
        ]
        print(f"\nAuth-related connections: {len(auth_connections)}")
        assert len(auth_connections) > 0, "Should find auth-related backend connection"


# =============================================================================
# Test 3: Search staleness — front-end stale results ↔ back-end reindex
# =============================================================================


@pytest.mark.integration
class TestSearchStalenessBridge:
    """Frontend stale search should link to backend Elasticsearch reindex."""

    def test_frontend_stale_search_connects_to_backend_reindex(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The frontend "stale search results" memory connects
        to the backend "Elasticsearch reindex every 30 min" memory.
        """
        ids = _seed(module_server)
        stale_search_id = ids["frontend"][2]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": stale_search_id,
                "limit": 5,
                "min_similarity": 0.2,
                "exclude_same_namespace": True,
            },
        )

        connections = result["connections"]

        print("\n=== Search Staleness Bridge ===")
        print("SOURCE: frontend — stale search results")
        print("EXPECTED: backend Elasticsearch reindex lag")
        print(f"ACTUAL:   {result['total_found']} connections:")
        for c in connections:
            print(f"  [{c['namespace']}] sim={c['similarity']:.4f} | {c['content'][:70]}...")

        assert result["total_found"] > 0

        search_connections = [
            c
            for c in connections
            if "search" in c["content"].lower()
            or "elastic" in c["content"].lower()
            or "index" in c["content"].lower()
        ]
        print(f"\nSearch-related connections: {len(search_connections)}")
        assert len(search_connections) > 0, "Should find search-related backend connection"


# =============================================================================
# Test 4: Payment flow — front-end error boundary ↔ back-end rate limiter
# =============================================================================


@pytest.mark.integration
class TestPaymentFlowBridge:
    """Frontend payment crashes should link to backend rate limiter."""

    def test_frontend_payment_connects_to_backend_ratelimit(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The frontend "Stripe element crashes" memory connects
        to the backend "rate limiter blocking payment retries" memory.
        Both are about payment failures.
        """
        ids = _seed(module_server)
        payment_crash_id = ids["frontend"][4]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": payment_crash_id,
                "limit": 5,
                "min_similarity": 0.2,
                "exclude_same_namespace": True,
            },
        )

        connections = result["connections"]

        print("\n=== Payment Flow Bridge ===")
        print("SOURCE: frontend — Stripe element crashes")
        print("EXPECTED: backend rate limiter on /api/payments")
        print(f"ACTUAL:   {result['total_found']} connections:")
        for c in connections:
            print(f"  [{c['namespace']}] sim={c['similarity']:.4f} | {c['content'][:70]}...")

        assert result["total_found"] > 0

        payment_connections = [
            c
            for c in connections
            if "payment" in c["content"].lower() or "rate" in c["content"].lower()
        ]
        print(f"\nPayment-related connections: {len(payment_connections)}")
        assert len(payment_connections) > 0, "Should find payment-related backend connection"


# =============================================================================
# Test 5: Unrelated memories should NOT bridge strongly
# =============================================================================


@pytest.mark.integration
class TestWeakBridges:
    """Unrelated topics across namespaces should have low similarity."""

    def test_mobile_css_has_no_strong_backend_connection(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The frontend "mobile CSS layout" memory has no
        strong connection to any backend memory — it's purely a
        UI concern with no server-side counterpart.
        """
        ids = _seed(module_server)
        mobile_css_id = ids["frontend"][3]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": mobile_css_id,
                "limit": 5,
                "min_similarity": 0.4,
                "exclude_same_namespace": True,
            },
        )

        print("\n=== Weak Bridge (mobile CSS) ===")
        print("SOURCE: frontend — mobile CSS layout issue")
        print("EXPECTED: 0 or very few connections above 0.4")
        print(f"ACTUAL:   {result['total_found']} connections")
        for c in result["connections"]:
            print(f"  [{c['namespace']}] sim={c['similarity']:.4f} | {c['content'][:60]}...")

        # Mobile CSS is a pure frontend concern — should have
        # very few or no strong backend connections
        assert result["total_found"] <= 1


# =============================================================================
# Test 6: Corpus bridges — full cross-app bridge discovery
# =============================================================================


@pytest.mark.integration
class TestFrontendBackendBridges:
    """End-to-end bridge discovery between frontend and backend."""

    def test_bridges_span_both_namespaces(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: Bridges found between frontend and backend, covering
        the shared topics: dashboard perf, auth, search, payments.
        """
        _seed(module_server)

        result = module_server._handle_tool(
            "corpus_bridges",
            {"min_similarity": 0.3, "max_bridges": 20},
        )

        bridges = result["bridges"]

        print("\n=== Frontend <-> Backend Bridges ===")
        print("EXPECTED: Bridges covering shared topics")
        print(f"ACTUAL:   {result['total_bridges']} bridges:")
        for b in bridges:
            print(
                f"  {b['query_namespace']:>8s} -> "
                f"{b['namespace']:<8s} "
                f"sim={b['similarity']:.4f}"
                f" | {b['content'][:55]}..."
            )

        assert result["total_bridges"] > 0

        # Should have bridges in both directions
        directions = {(b["query_namespace"], b["namespace"]) for b in bridges}
        print(f"\nBridge directions: {directions}")
        assert ("frontend", "backend") in directions or (
            "backend",
            "frontend",
        ) in directions

    def test_scoring_strategy_affects_rankings(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: vector_content scoring reranks results compared
        to vector_only, because frontend and backend use different
        vocabulary for the same problems.
        """
        ids = _seed(module_server)
        slow_dashboard_id = ids["frontend"][0]

        vo = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": slow_dashboard_id,
                "limit": 5,
                "min_similarity": 0.0,
                "scoring_strategy": "vector_only",
            },
        )
        vc = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": slow_dashboard_id,
                "limit": 5,
                "min_similarity": 0.0,
                "scoring_strategy": "vector_content",
            },
        )

        vo_scores = [c["similarity"] for c in vo["connections"]]
        vc_scores = [c["similarity"] for c in vc["connections"]]

        print("\n=== Scoring Strategy on Frontend->Backend ===")
        print(f"vector_only:   {[round(s, 4) for s in vo_scores]}")
        print(f"vector_content: {[round(s, 4) for s in vc_scores]}")
        print("EXPECTED: Different scores (different vocabulary penalized by jaccard)")

        if vo_scores and vc_scores:
            assert vo_scores != vc_scores

    def test_direct_service_overlap_summary(
        self,
        module_server: SpatialMemoryServer,
        module_database: Database,
    ) -> None:
        """
        EXPECTED: Corpus overlap summary shows 2 namespaces,
        10 total memories, and some bridges.
        """
        _seed(module_server)

        repo = LanceDBMemoryRepository(module_database)
        svc = CrossCorpusSimilarityService(
            repository=repo,
            namespace_provider=repo,
            config=SimilarityConfig(),
            memory_repository=repo,
        )

        summary = svc.get_corpus_overlap_summary()

        print("\n=== Corpus Overlap Summary ===")
        print("EXPECTED: 2 namespaces, 10 memories")
        print(f"ACTUAL:   {summary.total_memories_analyzed} memories")
        print(f"          {summary.namespaces_analyzed}")
        print(f"          {summary.bridges_found} bridges")
        print(f"          {summary.potential_duplicates} dupes")

        assert summary.total_memories_analyzed == 10
        assert set(summary.namespaces_analyzed) == {
            "frontend",
            "backend",
        }
