# Spatial Memory MCP Server — Architecture Diagrams

> Visual architecture reference for the Spatial Memory MCP Server

> **Status**: All components shown are fully implemented. See [README.md](README.md) for details.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph Clients["🤖 Any LLM Client"]
        Claude["Claude"]
        GPT["GPT-4/5"]
        Gemini["Gemini"]
        Llama["Llama/Mistral"]
        Local["Local LLMs"]
    end

    subgraph MCP["📡 MCP Protocol Layer"]
        Server["Spatial Memory<br/>MCP Server"]
    end

    subgraph Tools["🔧 Tool Categories"]
        direction LR
        Core["Core<br/>───────<br/>remember<br/>recall<br/>nearby<br/>forget"]
        Spatial["Spatial<br/>───────<br/>journey<br/>wander<br/>regions<br/>visualize"]
        Lifecycle["Lifecycle<br/>───────<br/>consolidate<br/>extract<br/>decay<br/>reinforce"]
        Utility["Utility<br/>───────<br/>stats<br/>namespaces<br/>export<br/>import"]
    end

    subgraph Services["⚙️ Service Layer"]
        MemService["Memory Service"]
        SpatialService["Spatial Service"]
        LifecycleService["Lifecycle Service"]
        VizService["Visualization Service"]
    end

    subgraph Embedding["🧠 Embedding Layer"]
        Router["Embedding Router"]
        LocalEmbed["Local Model<br/>(sentence-transformers)"]
        APIEmbed["API Model<br/>(OpenAI)"]
    end

    subgraph Storage["💾 Storage Layer"]
        LanceDB["LanceDB"]
        Memories[(memories table)]
        Meta[(metadata)]
    end

    Clients -->|"Tool Calls"| Server
    Server --> Core & Spatial & Lifecycle & Utility
    Core --> MemService
    Spatial --> SpatialService
    Lifecycle --> LifecycleService
    Utility --> MemService
    
    MemService & SpatialService --> Router
    Router --> LocalEmbed & APIEmbed
    
    MemService & SpatialService & LifecycleService & VizService --> LanceDB
    LanceDB --> Memories & Meta
```

---

## 2. Tool Ecosystem

```mermaid
mindmap
    root((Spatial Memory<br/>MCP Server))
        Core Operations
            remember
                Store memory
                Embed content
                Assign metadata
            remember_batch
                Bulk storage
                Efficient ingestion
            recall
                Semantic search
                Filter support
                Hybrid mode
            nearby
                K-nearest neighbors
                Spatial proximity
            forget
                Delete by ID
                Delete by filter
            forget_batch
                Bulk deletion
        Spatial Operations
            journey
                SLERP interpolation
                Path discovery
                Concept bridging
            wander
                Random walk
                Serendipity
                Exploration
            regions
                HDBSCAN clustering
                Auto-discovery
                Outlier detection
            visualize
                JSON coordinates
                Mermaid diagrams
                SVG rendering
        Lifecycle Operations
            consolidate
                Deduplication
                Merge strategies
                Similarity threshold
            extract
                Pattern matching
                Auto-capture
                Fact extraction
            decay
                Time-based
                Access-based
                Importance reduction
            reinforce
                Boost importance
                Track access
        Utility Operations
            stats
                Memory count
                Storage size
                Model info
            namespaces
                List/Create/Delete
                Multi-tenant
            export_memories
                JSON format
                Backup
            import_memories
                Restore
                Migration
```

---

## 3. Data Flow — Remember Operation

```mermaid
sequenceDiagram
    autonumber
    participant LLM as 🤖 LLM Client
    participant MCP as 📡 MCP Server
    participant MS as ⚙️ Memory Service
    participant ES as 🧠 Embedding Service
    participant DB as 💾 LanceDB

    LLM->>MCP: remember(content, metadata)
    MCP->>MS: store_memory(content, metadata)
    MS->>ES: embed(content)
    
    alt Local Model
        ES->>ES: sentence-transformers encode
    else API Model
        ES->>ES: OpenAI API call
    end
    
    ES-->>MS: vector[384]
    
    MS->>MS: Generate UUID
    MS->>MS: Add timestamps
    MS->>MS: Set importance
    
    MS->>DB: INSERT (id, content, vector, metadata)
    DB-->>MS: success
    
    MS-->>MCP: {id, content, created_at}
    MCP-->>LLM: Memory stored ✓
```

---

## 4. Data Flow — Recall Operation

```mermaid
sequenceDiagram
    autonumber
    participant LLM as 🤖 LLM Client
    participant MCP as 📡 MCP Server
    participant MS as ⚙️ Memory Service
    participant ES as 🧠 Embedding Service
    participant DB as 💾 LanceDB

    LLM->>MCP: recall(query, limit, filters)
    MCP->>MS: search_memories(query, limit, filters)
    MS->>ES: embed(query)
    ES-->>MS: query_vector[384]
    
    alt Pure Vector Search
        MS->>DB: vector_search(query_vector, limit)
    else Hybrid Search
        MS->>DB: vector_search(query_vector)
        MS->>DB: keyword_search(query)
        MS->>MS: RRF fusion
    end
    
    DB-->>MS: raw_results[]
    
    MS->>MS: Apply filters
    MS->>MS: Update access_count
    MS->>MS: Update last_accessed
    
    MS-->>MCP: memories[]
    MCP-->>LLM: [{id, content, similarity, metadata}...]
```

---

## 5. Data Flow — Journey Operation (SLERP)

```mermaid
sequenceDiagram
    autonumber
    participant LLM as 🤖 LLM Client
    participant MCP as 📡 MCP Server
    participant SS as 🧭 Spatial Service
    participant DB as 💾 LanceDB

    LLM->>MCP: journey(start_id, end_id, steps=5)
    MCP->>SS: create_journey(start_id, end_id, 5)
    
    SS->>DB: get_vector(start_id)
    DB-->>SS: start_vector
    SS->>DB: get_vector(end_id)
    DB-->>SS: end_vector
    
    loop For each step (0 to 5)
        SS->>SS: SLERP interpolate at t
        SS->>DB: nearest_neighbor(interpolated_vector)
        DB-->>SS: closest_memory
        SS->>SS: Record step + distance
    end
    
    SS-->>MCP: {path: [...], total_distance, density}
    MCP-->>LLM: Journey with discovered concepts
```

---

## 6. Data Flow — Regions Operation (Clustering)

```mermaid
sequenceDiagram
    autonumber
    participant LLM as 🤖 LLM Client
    participant MCP as 📡 MCP Server
    participant SS as 🧭 Spatial Service
    participant DB as 💾 LanceDB

    LLM->>MCP: regions(min_cluster_size=3)
    MCP->>SS: discover_regions(min_size=3)
    
    SS->>DB: get_all_vectors(namespace)
    DB-->>SS: vectors[], ids[], contents[]
    
    SS->>SS: HDBSCAN clustering
    Note over SS: Automatic cluster detection<br/>Outlier identification
    
    SS->>SS: Calculate centroids
    SS->>SS: Find centroid memories
    SS->>SS: Generate cluster labels
    SS->>SS: Compute silhouette score
    
    SS-->>MCP: {clusters: [...], outliers, quality}
    MCP-->>LLM: Discovered conceptual regions
```

---

## 7. Module Structure

```mermaid
flowchart TB
    subgraph Package["📦 spatial_memory"]
        Main["__main__.py<br/>Entry Point"]
        Server["server.py<br/>MCP Server"]
        Config["config.py<br/>Settings"]
        
        subgraph Core["core/"]
            Database["database.py<br/>LanceDB Wrapper"]
            Embeddings["embeddings.py<br/>Embedding Router"]
            Models["models.py<br/>Pydantic Models"]
            Errors["errors.py<br/>Exceptions"]
        end
        
        subgraph Services["services/"]
            MemorySvc["memory.py<br/>CRUD Operations"]
            SpatialSvc["spatial.py<br/>Journey/Wander/Regions"]
            LifecycleSvc["lifecycle.py<br/>Consolidate/Extract/Decay"]
            VizSvc["visualization.py<br/>JSON/Mermaid/SVG"]
        end
        
        subgraph ToolDefs["tools/"]
            CoreTools["core_tools.py"]
            SpatialTools["spatial_tools.py"]
            LifecycleTools["lifecycle_tools.py"]
            UtilityTools["utility_tools.py"]
        end
    end
    
    Main --> Server
    Server --> Config
    Server --> ToolDefs
    ToolDefs --> Services
    Services --> Core
    Core --> Config
```

---

## 8. Memory Data Model

```mermaid
erDiagram
    MEMORY {
        string id PK "UUID"
        string content "Text content"
        vector embedding "float[384]"
        timestamp created_at "Creation time"
        timestamp updated_at "Last modified"
        timestamp last_accessed "Last retrieval"
        int access_count "Times retrieved"
        float importance "0.0 - 1.0"
        string namespace "Tenant isolation"
        array tags "User-defined tags"
        string source "manual|extracted|consolidated"
        json metadata "Arbitrary data"
    }
    
    NAMESPACE {
        string name PK
        timestamp created_at
        int memory_count
    }
    
    NAMESPACE ||--o{ MEMORY : contains
```

---

## 9. Embedding Router Logic

```mermaid
flowchart TD
    Start([Embed Request]) --> Check{Config:<br/>embedding_model}
    
    Check -->|"all-MiniLM-L6-v2"| Local["Local: sentence-transformers"]
    Check -->|"all-mpnet-base-v2"| Local
    Check -->|"openai:text-embedding-3-*"| API["API: OpenAI"]
    Check -->|"openai:*"| API
    
    Local --> LoadModel{Model<br/>Loaded?}
    LoadModel -->|No| Download["Download from HuggingFace"]
    Download --> Cache["Cache locally"]
    Cache --> Encode
    LoadModel -->|Yes| Encode["model.encode(text)"]
    
    API --> APICall["POST /v1/embeddings"]
    APICall --> Parse["Parse response"]
    
    Encode --> Vector([Return vector])
    Parse --> Vector
```

---

## 10. Lifecycle — Decay Algorithm

```mermaid
flowchart TD
    Start([decay trigger]) --> GetAll["Get all memories"]
    GetAll --> Loop{For each<br/>memory}
    
    Loop --> CalcAge["Calculate days_since_access"]
    CalcAge --> CheckThreshold{days ><br/>threshold?}
    
    CheckThreshold -->|No| Skip["Skip - recently accessed"]
    CheckThreshold -->|Yes| CalcDecay["Calculate decay amount"]
    
    CalcDecay --> Strategy{Strategy?}
    
    Strategy -->|time| TimeDecay["decay = rate × days_factor"]
    Strategy -->|access| AccessDecay["decay = rate × (1/access_count)"]
    Strategy -->|combined| CombinedDecay["decay = rate × (time + access)/2"]
    
    TimeDecay & AccessDecay & CombinedDecay --> ApplyDecay["new_importance = old - decay"]
    
    ApplyDecay --> CheckFloor{Below<br/>min_importance?}
    CheckFloor -->|Yes| SetFloor["Set to min_importance"]
    CheckFloor -->|No| Update["Update memory"]
    SetFloor --> Update
    
    Update --> Loop
    Skip --> Loop
    
    Loop -->|Done| Return([Return affected count])
```

---

## 11. Visualization Output Formats

```mermaid
flowchart LR
    subgraph Input
        Memories[(Memory<br/>Vectors)]
    end
    
    subgraph Processing
        UMAP["UMAP<br/>Dimensionality<br/>Reduction"]
        Cluster["HDBSCAN<br/>Clustering"]
    end
    
    subgraph Output
        JSON["JSON<br/>─────────<br/>{nodes, edges,<br/>clusters, bounds}"]
        Mermaid["Mermaid<br/>─────────<br/>mindmap or<br/>flowchart syntax"]
        SVG["SVG<br/>─────────<br/>Rendered scatter<br/>plot image"]
    end
    
    Memories --> UMAP
    UMAP --> Cluster
    Cluster --> JSON & Mermaid & SVG
```

---

## 12. Comparison: Official MCP Memory vs Ours

```mermaid
flowchart TB
    subgraph Official["Official MCP Memory Server"]
        direction TB
        OModel["Knowledge Graph Model"]
        OStore["JSONL File Storage"]
        OSearch["Keyword Search"]
        OTools["9 Tools:<br/>create_entities, create_relations,<br/>add_observations, delete_*, read_graph,<br/>search_nodes, open_nodes"]
    end
    
    subgraph Ours["Spatial Memory MCP Server"]
        direction TB
        SModel["Vector Space Model"]
        SStore["LanceDB Storage"]
        SSearch["Semantic + Hybrid Search"]
        STools["21 Tools:<br/>Core (6) + Spatial (4) +<br/>Lifecycle (4) + Utility (7)"]
        SExtra["+ Visualization<br/>+ Auto-clustering<br/>+ Path interpolation<br/>+ Memory dynamics"]
    end
    
    Official -.->|"Finds"| Keyword["Exact keyword<br/>matches only"]
    Ours -.->|"Finds"| Semantic["Semantically related<br/>concepts"]
    
    style Ours fill:#1a4d1a
    style SExtra fill:#2d5a2d
```

---

## 13. Integration with RLM Pattern

```mermaid
flowchart TB
    subgraph RLM["RLM Orchestrator (Parent LLM)"]
        Root["Root LLM<br/>(Clean Context)"]
        REPL["Python REPL<br/>Environment"]
    end
    
    subgraph SubCalls["Sub-LLM Sandboxes"]
        Sub1["Sub-LLM 1"]
        Sub2["Sub-LLM 2"]
        Sub3["Sub-LLM N"]
    end
    
    subgraph Memory["Spatial Memory MCP"]
        Recall["recall()"]
        Journey["journey()"]
        Regions["regions()"]
        Viz["visualize()"]
    end
    
    Root --> REPL
    REPL -->|"Spawns"| Sub1 & Sub2 & Sub3
    
    Root -->|"Direct tool use"| Memory
    Sub1 & Sub2 & Sub3 -->|"Query memory"| Memory
    
    Memory -->|"Concise results"| REPL
    Sub1 & Sub2 & Sub3 -->|"Summaries"| REPL
    REPL -->|"Aggregated answer"| Root
    
    style Memory fill:#1a3d5c
```

---

## 14. Development Timeline

```mermaid
gantt
    title Spatial Memory MCP Server - Development Phases
    dateFormat  YYYY-MM-DD
    
    section Phase 1: Foundation
    Project setup           :p1a, 2026-01-20, 2d
    Config system          :p1b, after p1a, 2d
    LanceDB integration    :p1c, after p1b, 2d
    Embedding service      :p1d, after p1c, 1d
    
    section Phase 2: Core Ops
    remember/forget        :p2a, after p1d, 2d
    recall with filters    :p2b, after p2a, 2d
    nearby                 :p2c, after p2b, 1d
    MCP server setup       :p2d, after p2c, 2d
    
    section Phase 3: Spatial
    SLERP + journey        :p3a, after p2d, 2d
    wander                 :p3b, after p3a, 2d
    HDBSCAN + regions      :p3c, after p3b, 2d
    visualize (JSON)       :p3d, after p3c, 1d
    
    section Phase 4: Lifecycle
    consolidate            :p4a, after p3d, 2d
    extract                :p4b, after p4a, 2d
    decay + reinforce      :p4c, after p4b, 2d
    visualize (Mermaid/SVG):p4d, after p4c, 1d
    
    section Phase 5: Utilities
    stats + namespaces     :p5a, after p4d, 2d
    export/import          :p5b, after p5a, 2d
    hybrid search          :p5c, after p5b, 2d
    OpenAI embeddings      :p5d, after p5c, 1d
    
    section Phase 6: Polish
    Integration tests      :p6a, after p5d, 3d
    Documentation          :p6b, after p6a, 2d
    PyPI release           :p6c, after p6b, 2d
```

---

*Generated for Spatial Memory MCP Server v0.1.0*
