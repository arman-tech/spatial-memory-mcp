"""Manual integration tests for the Cross-Corpus Similarity Engine.

Run with:
    pytest tests/integration/test_cross_corpus_integration.py -v -s -m ""

The -s flag shows print output so you can visually verify results.
Each test prints EXPECTED vs ACTUAL comparisons.
"""

from __future__ import annotations

import pytest

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.core.models import SimilarityConfig
from spatial_memory.server import SpatialMemoryServer
from spatial_memory.services.similarity import CrossCorpusSimilarityService

# =============================================================================
# Test Data: Memories across 3 namespaces simulating 3 apps
# =============================================================================

ALL_MEMORIES = [
    # Namespace "slack" — communication-related memories
    {
        "content": (
            "Team decided to migrate the authentication system"
            " from session-based to JWT tokens for better"
            " microservice compatibility"
        ),
        "namespace": "slack",
        "tags": ["auth", "jwt", "architecture", "decision"],
        "importance": 0.9,
    },
    {
        "content": (
            "Daily standup: frontend team blocked on API"
            " changes, backend deploying Redis cache today"
        ),
        "namespace": "slack",
        "tags": ["standup", "frontend", "redis"],
        "importance": 0.3,
    },
    {
        "content": (
            "Bug report: users getting 401 errors after token"
            " expiry, need to implement refresh token flow"
        ),
        "namespace": "slack",
        "tags": ["bug", "auth", "jwt", "tokens"],
        "importance": 0.8,
    },
    # Namespace "jira" — project management memories
    {
        "content": (
            "JIRA-1234: Implement JWT authentication with refresh tokens for the API gateway"
        ),
        "namespace": "jira",
        "tags": ["auth", "jwt", "api-gateway", "ticket"],
        "importance": 0.8,
    },
    {
        "content": ("JIRA-1235: Set up Redis caching layer for frequently accessed user profiles"),
        "namespace": "jira",
        "tags": ["redis", "caching", "performance", "ticket"],
        "importance": 0.7,
    },
    {
        "content": ("JIRA-1236: Migrate database from PostgreSQL 14 to PostgreSQL 16"),
        "namespace": "jira",
        "tags": ["database", "postgresql", "migration", "ticket"],
        "importance": 0.6,
    },
    # Namespace "ide" — code-related memories from the IDE
    {
        "content": (
            "Implemented JWT token validation middleware"
            " with RS256 signing and automatic key rotation"
        ),
        "namespace": "ide",
        "tags": ["auth", "jwt", "middleware", "code"],
        "importance": 0.9,
    },
    {
        "content": ("Added Redis connection pooling with sentinel support for high availability"),
        "namespace": "ide",
        "tags": ["redis", "connection-pool", "ha", "code"],
        "importance": 0.7,
    },
    {
        "content": ("Wrote unit tests for the user registration endpoint with email validation"),
        "namespace": "ide",
        "tags": ["testing", "registration", "email", "code"],
        "importance": 0.5,
    },
]


def _store_all_memories(
    server: SpatialMemoryServer,
) -> dict[str, list[str]]:
    """Store all test memories and return IDs grouped by namespace."""
    ids: dict[str, list[str]] = {"slack": [], "jira": [], "ide": []}
    for mem in ALL_MEMORIES:
        result = server._handle_tool("remember", mem)
        ids[mem["namespace"]].append(result["id"])
    return ids


# =============================================================================
# Test 1: Verify memories stored across 3 namespaces
# =============================================================================


@pytest.mark.integration
class TestSetupVerification:
    """Verify the test data is stored correctly."""

    def test_memories_stored_in_three_namespaces(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: 3 namespaces (slack, jira, ide) with 3 memories each.
        """
        ids = _store_all_memories(module_server)
        result = module_server._handle_tool("namespaces", {})
        ns_names = [ns["name"] for ns in result["namespaces"]]

        print("\n=== TEST 1: Setup Verification ===")
        print("EXPECTED: 3 namespaces: slack, jira, ide")
        print(f"ACTUAL:   {len(ns_names)} namespaces: {sorted(ns_names)}")
        print("EXPECTED: 3 memories per namespace")
        for ns in ["slack", "jira", "ide"]:
            print(f"  {ns}: {len(ids[ns])} memories")

        assert set(ns_names) == {"slack", "jira", "ide"}
        assert all(len(v) == 3 for v in ids.values())


# =============================================================================
# Test 2: discover_connections — JWT auth memory should bridge all 3 apps
# =============================================================================


@pytest.mark.integration
class TestDiscoverConnections:
    """Test the discover_connections MCP tool."""

    def test_jwt_memory_finds_cross_namespace_matches(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The Slack JWT decision memory should find related JWT
        memories in jira (JIRA-1234) and ide (JWT middleware).
        """
        ids = _store_all_memories(module_server)
        jwt_memory_id = ids["slack"][0]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": jwt_memory_id,
                "limit": 5,
                "min_similarity": 0.3,
            },
        )

        connections = result["connections"]

        print("\n=== TEST 2a: discover_connections (JWT) ===")
        print(f"Source: Slack JWT decision (id={jwt_memory_id[:8]}...)")
        print("EXPECTED: Connections in jira and ide namespaces about JWT/auth")
        print(f"ACTUAL:   {result['total_found']} connections found:")
        for c in connections:
            print(f"  [{c['namespace']:5s}] sim={c['similarity']:.4f} | {c['content'][:70]}...")

        assert result["total_found"] > 0
        connected_ns = {c["namespace"] for c in connections}
        print("\nEXPECTED: Connections from jira and/or ide")
        print(f"ACTUAL:   Connections from: {connected_ns}")

        non_slack = [c for c in connections if c["namespace"] != "slack"]
        assert len(non_slack) > 0, "Should find connections outside slack"

    def test_exclude_same_namespace(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: With exclude_same_namespace=True, no slack results.
        """
        ids = _store_all_memories(module_server)
        jwt_memory_id = ids["slack"][0]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": jwt_memory_id,
                "limit": 10,
                "min_similarity": 0.0,
                "exclude_same_namespace": True,
            },
        )

        connections = result["connections"]

        print("\n=== TEST 2b: exclude_same_namespace ===")
        print("EXPECTED: No connections from 'slack' namespace")
        namespaces_found = {c["namespace"] for c in connections}
        print(f"ACTUAL:   Namespaces in results: {namespaces_found}")

        assert "slack" not in namespaces_found

    def test_unrelated_memory_has_fewer_connections(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: The user registration test memory (ide[2]) should
        have fewer high-similarity connections than the JWT memory.
        """
        ids = _store_all_memories(module_server)
        registration_id = ids["ide"][2]

        result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": registration_id,
                "limit": 5,
                "min_similarity": 0.5,
            },
        )

        print("\n=== TEST 2c: Unrelated memory (registration) ===")
        print("EXPECTED: Fewer connections above 0.5 similarity")
        print(f"ACTUAL:   {result['total_found']} connections")
        for c in result["connections"]:
            print(f"  [{c['namespace']:5s}] sim={c['similarity']:.4f} | {c['content'][:60]}...")


# =============================================================================
# Test 3: corpus_bridges — should find JWT and Redis bridges
# =============================================================================


@pytest.mark.integration
class TestCorpusBridges:
    """Test the corpus_bridges MCP tool."""

    def test_finds_bridges_across_namespaces(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: Bridges found between namespaces, especially:
        - JWT/auth topic: slack <-> jira <-> ide
        - Redis topic: slack <-> jira <-> ide
        """
        _store_all_memories(module_server)

        result = module_server._handle_tool(
            "corpus_bridges",
            {"min_similarity": 0.3, "max_bridges": 20},
        )

        bridges = result["bridges"]

        print("\n=== TEST 3a: corpus_bridges ===")
        print("EXPECTED: Multiple bridges (JWT and Redis topics)")
        print(f"ACTUAL:   {result['total_bridges']} bridges found:")
        for b in bridges:
            print(
                f"  {b['query_namespace']:5s} -> "
                f"{b['namespace']:5s} "
                f"sim={b['similarity']:.4f}"
                f" | {b['content'][:60]}..."
            )

        assert result["total_bridges"] > 0

        bridge_pairs = {(b["query_namespace"], b["namespace"]) for b in bridges}
        print(f"\nNamespace pairs bridged: {bridge_pairs}")

    def test_high_similarity_threshold_filters(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: Higher threshold returns fewer bridges.
        """
        _store_all_memories(module_server)

        low_result = module_server._handle_tool(
            "corpus_bridges",
            {"min_similarity": 0.3, "max_bridges": 50},
        )
        high_result = module_server._handle_tool(
            "corpus_bridges",
            {"min_similarity": 0.7, "max_bridges": 50},
        )

        print("\n=== TEST 3b: Threshold filtering ===")
        print("EXPECTED: High threshold returns fewer bridges")
        lo = low_result["total_bridges"]
        hi = high_result["total_bridges"]
        print(f"  min_similarity=0.3: {lo} bridges")
        print(f"  min_similarity=0.7: {hi} bridges")

        assert lo >= hi

    def test_namespace_filter(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: Filtering to slack+jira should exclude ide.
        """
        _store_all_memories(module_server)

        result = module_server._handle_tool(
            "corpus_bridges",
            {
                "min_similarity": 0.3,
                "max_bridges": 50,
                "namespace_filter": ["slack", "jira"],
            },
        )

        bridges = result["bridges"]
        namespaces_in_results = set()
        for b in bridges:
            namespaces_in_results.add(b["namespace"])
            namespaces_in_results.add(b["query_namespace"])

        print("\n=== TEST 3c: Namespace filter (slack+jira) ===")
        print("EXPECTED: No 'ide' in results")
        print(f"ACTUAL:   Namespaces: {namespaces_in_results}")
        print(f"          {result['total_bridges']} bridges")

        assert "ide" not in namespaces_in_results


# =============================================================================
# Test 4: Scoring strategy differences
# =============================================================================


@pytest.mark.integration
class TestScoringStrategies:
    """Verify different scoring strategies produce different rankings."""

    def test_vector_content_differs_from_vector_only(
        self, module_server: SpatialMemoryServer
    ) -> None:
        """
        EXPECTED: vector_content strategy produces different scores
        than vector_only because of the lexical jaccard component.
        """
        ids = _store_all_memories(module_server)
        jwt_memory_id = ids["slack"][0]

        vo_result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": jwt_memory_id,
                "limit": 5,
                "min_similarity": 0.0,
                "scoring_strategy": "vector_only",
            },
        )
        vc_result = module_server._handle_tool(
            "discover_connections",
            {
                "memory_id": jwt_memory_id,
                "limit": 5,
                "min_similarity": 0.0,
                "scoring_strategy": "vector_content",
            },
        )

        print("\n=== TEST 4: Scoring strategy comparison ===")
        print("vector_only results:")
        for c in vo_result["connections"]:
            print(f"  sim={c['similarity']:.4f} | {c['content'][:60]}...")
        print("vector_content results:")
        for c in vc_result["connections"]:
            print(f"  sim={c['similarity']:.4f} | {c['content'][:60]}...")

        vo_scores = [c["similarity"] for c in vo_result["connections"]]
        vc_scores = [c["similarity"] for c in vc_result["connections"]]
        print(f"\nvector_only  scores: {[round(s, 4) for s in vo_scores]}")
        print(f"vector_content scores: {[round(s, 4) for s in vc_scores]}")

        if vo_scores and vc_scores:
            assert vo_scores != vc_scores, "Scores should differ between strategies"


# =============================================================================
# Test 5: Service-level API (direct Python, no MCP)
# =============================================================================


@pytest.mark.integration
class TestServiceDirectAPI:
    """Test CrossCorpusSimilarityService directly (bypassing MCP)."""

    def _make_service(self, db: Database) -> CrossCorpusSimilarityService:
        repo = LanceDBMemoryRepository(db)
        return CrossCorpusSimilarityService(
            repository=repo,
            namespace_provider=repo,
            config=SimilarityConfig(),
            memory_repository=repo,
        )

    def test_find_similar_to_memory(
        self,
        module_server: SpatialMemoryServer,
        module_database: Database,
    ) -> None:
        """
        EXPECTED: find_similar_to_memory returns CrossCorpusMatch
        objects with all fields populated.
        """
        ids = _store_all_memories(module_server)
        svc = self._make_service(module_database)
        jwt_id = ids["slack"][0]

        matches = svc.find_similar_to_memory(jwt_id, limit=3, min_similarity=0.0)

        print("\n=== TEST 5a: Direct service API ===")
        print(f"find_similar_to_memory('{jwt_id[:8]}...')")
        print("EXPECTED: CrossCorpusMatch objects with fields")
        for m in matches:
            print(
                f"  id={m.memory_id[:8]}... ns={m.namespace} "
                f"sim={m.similarity:.4f} "
                f"raw={m.raw_vector_similarity:.4f} "
                f"strategy={m.scoring_strategy}"
            )
            print(f"    content: {m.content[:60]}...")
            print(f"    tags: {m.tags}")

        assert len(matches) > 0
        for m in matches:
            assert m.memory_id != jwt_id, "Should exclude self"
            assert m.scoring_strategy == "vector_only"
            assert 0.0 <= m.similarity <= 1.0
            assert m.namespace in ("slack", "jira", "ide")

    def test_find_similar_batch(
        self,
        module_server: SpatialMemoryServer,
        module_database: Database,
        session_embedding_service: EmbeddingService,
    ) -> None:
        """
        EXPECTED: Batch query returns parallel results for 2 vectors.
        """
        import numpy as np

        _store_all_memories(module_server)
        svc = self._make_service(module_database)

        v1 = session_embedding_service.embed("JWT authentication tokens")
        v2 = session_embedding_service.embed("Redis caching performance")

        results = svc.find_similar_batch(
            [np.array(v1), np.array(v2)],
            limit_per_query=3,
            min_similarity=0.0,
        )

        print("\n=== TEST 5b: Batch similarity ===")
        print("EXPECTED: 2 result sets, one per query vector")
        print(f"ACTUAL:   {len(results)} result sets")
        for i, batch in enumerate(results):
            query_desc = ["JWT auth", "Redis caching"][i]
            print(f"\n  Query {i} ({query_desc}):")
            print(f"    {len(batch.matches)} matches:")
            for m in batch.matches:
                print(f"      [{m.namespace:5s}] sim={m.similarity:.4f} | {m.content[:50]}...")

        assert len(results) == 2
        jwt_matches = results[0].matches
        assert any("jwt" in m.content.lower() or "auth" in m.content.lower() for m in jwt_matches)

    def test_get_corpus_overlap_summary(
        self,
        module_server: SpatialMemoryServer,
        module_database: Database,
    ) -> None:
        """
        EXPECTED: Summary shows all 3 namespaces with bridge counts.
        """
        _store_all_memories(module_server)
        svc = self._make_service(module_database)

        summary = svc.get_corpus_overlap_summary()

        print("\n=== TEST 5c: Corpus overlap summary ===")
        print("EXPECTED: 3 namespaces, 9 total memories, some bridges")
        print("ACTUAL:")
        print(f"  Total analyzed: {summary.total_memories_analyzed}")
        print(f"  Namespaces:     {summary.namespaces_analyzed}")
        print(f"  Bridges found:  {summary.bridges_found}")
        print(f"  Potential dupes: {summary.potential_duplicates}")
        print("  Top bridges:")
        for b in summary.top_bridges[:5]:
            print(
                f"    {b.query_namespace} -> {b.namespace} "
                f"sim={b.similarity:.4f}"
                f" | {b.content[:50]}..."
            )

        assert summary.total_memories_analyzed == 9
        assert set(summary.namespaces_analyzed) == {
            "slack",
            "jira",
            "ide",
        }


# =============================================================================
# Test 6: Edge cases
# =============================================================================


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nonexistent_memory_id(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: discover_connections with bad ID raises an error.
        """
        print("\n=== TEST 6a: Edge case — nonexistent memory ===")
        try:
            module_server._handle_tool(
                "discover_connections",
                {"memory_id": "nonexistent-id-12345"},
            )
            print("ACTUAL: No error raised (unexpected)")
            pytest.fail("Should have raised an error")
        except Exception as e:
            print("EXPECTED: Error about memory not found")
            print(f"ACTUAL:   {type(e).__name__}: {e}")

    def test_very_high_threshold_returns_empty(self, module_server: SpatialMemoryServer) -> None:
        """
        EXPECTED: Very high similarity threshold returns no bridges.
        """
        _store_all_memories(module_server)

        result = module_server._handle_tool(
            "corpus_bridges",
            {"min_similarity": 0.99, "max_bridges": 50},
        )

        print("\n=== TEST 6b: Very high threshold (0.99) ===")
        print("EXPECTED: 0 or very few bridges")
        print(f"ACTUAL:   {result['total_bridges']} bridges")
