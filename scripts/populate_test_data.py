"""Populate database with realistic test data for real-world testing.

Run: python scripts/populate_test_data.py

Creates memories across multiple namespaces with varied content.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from spatial_memory.adapters.lancedb_repository import LanceDBMemoryRepository
from spatial_memory.core.database import Database
from spatial_memory.core.embeddings import EmbeddingService
from spatial_memory.services.memory import MemoryService


# Realistic test data organized by namespace
TEST_DATA = {
    "architecture": [
        ("Use repository pattern to abstract data access layer", ["patterns", "data"], 0.9),
        ("Prefer composition over inheritance for flexibility", ["patterns", "oop"], 0.85),
        ("Apply SOLID principles: Single Responsibility, Open/Closed, Liskov, Interface Segregation, Dependency Inversion", ["patterns", "solid"], 0.95),
        ("Use dependency injection for testability and loose coupling", ["patterns", "testing"], 0.85),
        ("Implement circuit breakers for resilient microservices", ["microservices", "resilience"], 0.8),
        ("Event sourcing stores state as a sequence of events", ["patterns", "events"], 0.75),
        ("CQRS separates read and write models for scalability", ["patterns", "cqrs"], 0.8),
        ("Domain-driven design aligns code with business concepts", ["patterns", "ddd"], 0.85),
    ],
    "frontend": [
        ("React components should be small and focused", ["react", "components"], 0.8),
        ("Use React hooks for state management in functional components", ["react", "hooks"], 0.85),
        ("TypeScript catches errors at compile time", ["typescript", "types"], 0.9),
        ("Lazy loading improves initial page load performance", ["performance", "loading"], 0.75),
        ("CSS-in-JS provides scoped styles and dynamic theming", ["css", "styling"], 0.7),
        ("Virtual DOM enables efficient UI updates", ["react", "performance"], 0.8),
        ("Server-side rendering improves SEO and initial load", ["ssr", "nextjs"], 0.75),
        ("Web Vitals: LCP, FID, CLS are key performance metrics", ["performance", "metrics"], 0.8),
    ],
    "backend": [
        ("Use async/await for non-blocking I/O operations", ["python", "async"], 0.85),
        ("Connection pooling reduces database connection overhead", ["database", "performance"], 0.8),
        ("Rate limiting protects APIs from abuse", ["api", "security"], 0.9),
        ("Use structured logging for better observability", ["logging", "observability"], 0.75),
        ("Implement graceful shutdown for zero-downtime deployments", ["deployment", "reliability"], 0.8),
        ("Cache frequently accessed data in Redis", ["caching", "redis"], 0.85),
        ("Use message queues for async processing", ["messaging", "rabbitmq"], 0.75),
        ("Database transactions ensure data consistency", ["database", "transactions"], 0.85),
    ],
    "devops": [
        ("Docker containers provide consistent environments", ["docker", "containers"], 0.85),
        ("Kubernetes orchestrates container deployment and scaling", ["kubernetes", "orchestration"], 0.9),
        ("CI/CD pipelines automate testing and deployment", ["cicd", "automation"], 0.9),
        ("Infrastructure as Code with Terraform ensures reproducibility", ["terraform", "iac"], 0.85),
        ("Blue-green deployments enable zero-downtime releases", ["deployment", "strategy"], 0.8),
        ("Prometheus and Grafana for metrics and monitoring", ["monitoring", "observability"], 0.8),
        ("GitOps uses Git as single source of truth for infrastructure", ["gitops", "automation"], 0.75),
        ("Feature flags allow gradual rollout of changes", ["deployment", "flags"], 0.7),
    ],
    "security": [
        ("JWT tokens are stateless and good for distributed systems", ["authentication", "jwt"], 0.9),
        ("Always validate and sanitize user input", ["validation", "xss"], 0.95),
        ("Use parameterized queries to prevent SQL injection", ["database", "injection"], 0.95),
        ("HTTPS encrypts data in transit", ["encryption", "tls"], 0.9),
        ("Hash passwords with bcrypt or Argon2", ["passwords", "hashing"], 0.9),
        ("Implement CORS properly for API security", ["api", "cors"], 0.8),
        ("Rate limiting prevents brute force attacks", ["api", "dos"], 0.85),
        ("Principle of least privilege for access control", ["authorization", "rbac"], 0.85),
    ],
    "database": [
        ("Indexing dramatically improves query performance", ["indexing", "performance"], 0.9),
        ("Normalize data to reduce redundancy, denormalize for read performance", ["design", "normalization"], 0.85),
        ("Use database migrations for schema version control", ["migrations", "versioning"], 0.8),
        ("PostgreSQL excels at complex queries and data integrity", ["postgresql", "sql"], 0.85),
        ("MongoDB is good for flexible document schemas", ["mongodb", "nosql"], 0.75),
        ("Redis provides sub-millisecond latency for caching", ["redis", "caching"], 0.85),
        ("Database replication provides high availability", ["replication", "ha"], 0.8),
        ("Partitioning helps manage large tables", ["partitioning", "scaling"], 0.75),
    ],
    "testing": [
        ("Unit tests verify individual components in isolation", ["unit", "isolation"], 0.85),
        ("Integration tests verify component interactions", ["integration", "e2e"], 0.85),
        ("Mock external dependencies for reliable tests", ["mocking", "isolation"], 0.8),
        ("Test coverage should focus on critical paths", ["coverage", "quality"], 0.75),
        ("Property-based testing finds edge cases automatically", ["property", "fuzzing"], 0.7),
        ("TDD: write tests before implementation", ["tdd", "methodology"], 0.75),
        ("Snapshot testing catches unintended UI changes", ["snapshot", "react"], 0.7),
        ("Load testing validates performance under stress", ["load", "performance"], 0.8),
    ],
    "project-notes": [
        ("The spatial memory MCP server uses LanceDB for vector storage", ["spatial-memory", "lancedb"], 0.9),
        ("Embeddings are generated using sentence-transformers", ["spatial-memory", "embeddings"], 0.85),
        ("21 tools across 4 categories: core, spatial, lifecycle, utility", ["spatial-memory", "tools"], 0.9),
        ("Journey operation uses SLERP for path interpolation", ["spatial-memory", "spatial"], 0.8),
        ("Consolidate merges similar memories to reduce redundancy", ["spatial-memory", "lifecycle"], 0.8),
        ("Export supports Parquet, JSON, and CSV formats", ["spatial-memory", "export"], 0.75),
        ("Hybrid recall combines vector and full-text search", ["spatial-memory", "search"], 0.85),
        ("Path security prevents directory traversal attacks", ["spatial-memory", "security"], 0.9),
    ],
}


def populate_database(db_path: str = "./.spatial-memory") -> None:
    """Populate database with test data."""

    print("\n" + "=" * 60)
    print("POPULATING TEST DATA")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")

    # Initialize services
    print("\nInitializing services...")
    embeddings = EmbeddingService(model_name="all-MiniLM-L6-v2")
    db = Database(
        storage_path=db_path,
        embedding_dim=embeddings.dimensions,
        auto_create_indexes=True,
    )
    db.connect()
    repository = LanceDBMemoryRepository(db)
    service = MemoryService(repository=repository, embeddings=embeddings)

    total_created = 0

    for namespace, memories in TEST_DATA.items():
        print(f"\n  Namespace: {namespace}")

        # Use batch insert for efficiency
        batch = []
        for content, tags, importance in memories:
            batch.append({
                "content": content,
                "namespace": namespace,
                "tags": tags,
                "importance": importance,
            })

        result = service.remember_batch(batch)
        print(f"    Created {result.count} memories")
        total_created += result.count

    db.close()

    print(f"\n{'=' * 60}")
    print(f"COMPLETE: Created {total_created} memories across {len(TEST_DATA)} namespaces")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./.spatial-memory"
    populate_database(db_path)


if __name__ == "__main__":
    main()
