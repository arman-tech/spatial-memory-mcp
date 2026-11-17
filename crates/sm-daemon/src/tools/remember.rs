//! `remember` (SPEC 8) and the store path every new memory takes: redact,
//! derive the title, chunk with no index cap, insert with pending chunks,
//! wake the bulk queue, journal `chunks_per_memory`.
//!
//! Chunks are stored without embeddings and embedded by the bulk queue
//! (B-27), so `remember` never waits on the model's bulk session. A
//! memory is never stored without chunks: the chunker must return at least
//! one piece.

use std::collections::BTreeMap;

use serde_json::Value;
use sm_core::error::{CoreError, Result};
use sm_core::journal::metrics;
use sm_core::protocol::{RememberItem, RememberParams, RememberReply, Remembered};
use sm_core::traits::{ChunkPlan, SecretScanner};
use sm_core::types::{NewMemory, Provenance, ProvenanceSource, derive_title};
use sm_store::{Access, Inserted, MemoryInput, NewChunk};

use crate::dispatch::Ctx;
use crate::state::{ProjectEntry, Ready, Shared};

/// Chunks `content` for storage, pending embedding. `index_cap` is `None`
/// for explicit tools and `capture.max_index_tokens` for capture.
pub(crate) fn pending_chunks(
    sh: &Shared,
    ready: &Ready,
    content: &str,
    index_cap: Option<usize>,
) -> Result<(Vec<NewChunk>, ChunkPlan)> {
    let plan = ready.chunker.chunk(content, &sh.chunk_params(index_cap))?;
    if plan.pieces.is_empty() {
        return Err(CoreError::invalid_input(
            "the content gave no chunks to index; a memory is never stored without chunks",
        ));
    }
    let chunks = plan.pieces.iter().cloned().map(NewChunk::pending).collect();
    Ok((chunks, plan))
}

/// A memory from one `remember` item, with the redaction kinds.
pub(crate) fn prepare_item(
    ctx: &Ctx<'_>,
    ready: &Ready,
    project: &ProjectEntry,
    item: &RememberItem,
) -> Result<(MemoryInput, Vec<String>)> {
    let sh = ctx.sh;
    let content = sh.scanner.redact(&item.content);
    let mut kinds = content.kinds;
    let title = match &item.title {
        Some(t) => {
            let r = sh.scanner.redact(t);
            kinds.extend(r.kinds);
            Some(r.text)
        }
        None => derive_title(&content.text),
    };
    let mut provenance = Provenance::new(ProvenanceSource::Agent);
    provenance.author.clone_from(&item.author);
    provenance.session_id = ctx.session().map(str::to_owned);
    provenance.file_path.clone_from(&item.file_path);
    provenance.commit_sha = project.identity.repo.as_ref().and_then(|r| r.head_commit());
    provenance.tool = Some("remember".to_owned());
    let (chunks, _) = pending_chunks(sh, ready, &content.text, None)?;
    let mut memory = NewMemory::new(project.project.id, content.text, provenance);
    memory.title = title;
    if let Some(t) = item.memory_type {
        memory.memory_type = t;
    }
    if let Some(i) = item.importance {
        memory.importance = i;
    }
    memory.tags.clone_from(&item.tags);
    memory.anchors.clone_from(&item.anchors);
    memory.supersedes_id = item.supersedes;
    Ok((MemoryInput { memory, chunks }, kinds))
}

/// Inserts under local access, wakes the queue and journals
/// `chunks_per_memory` for each memory created.
pub(crate) fn store(
    sh: &Shared,
    request_id: &str,
    project: &str,
    inputs: Vec<MemoryInput>,
) -> Result<Vec<Inserted>> {
    let inserted = sh.store.insert_memories(inputs, Access::Local)?;
    sh.after_write();
    for i in &inserted {
        if i.chunks > 0 {
            let mut labels = BTreeMap::new();
            labels.insert("project".to_owned(), Value::from(project));
            sh.journal.record(
                request_id,
                metrics::CHUNKS_PER_MEMORY,
                f64::from(i.chunks),
                labels,
            );
        }
    }
    Ok(inserted)
}

pub(crate) fn remember(ctx: &Ctx<'_>, p: &RememberParams) -> Result<RememberReply> {
    let ready = ctx.sh.ready()?;
    let project = ctx.project()?;
    let mut inputs = Vec::with_capacity(p.memories.len());
    let mut redactions = Vec::with_capacity(p.memories.len());
    for item in &p.memories {
        let (input, kinds) = prepare_item(ctx, &ready, &project, item)?;
        inputs.push(input);
        redactions.push(kinds);
    }
    let name = project.project.name.clone();
    let inserted = store(ctx.sh, &ctx.req.id, &name, inputs)?;
    Ok(RememberReply {
        memories: inserted
            .into_iter()
            .zip(redactions)
            .map(|(i, r)| Remembered {
                id: i.id,
                project: name.clone(),
                outcome: i.outcome,
                redactions: r,
                chunks: i.chunks,
                superseded: i.superseded,
            })
            .collect(),
    })
}
