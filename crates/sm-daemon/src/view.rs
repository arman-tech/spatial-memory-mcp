//! `MemoryView`: how every tool shows a memory (SPEC 8).

use std::collections::HashMap;

use sm_core::error::Result;
use sm_core::ids::ProjectId;
use sm_core::protocol::MemoryView;
use sm_core::types::{Memory, SourceTag};
use sm_store::Store;

/// Project names by id, read once per request.
pub(crate) struct Names(HashMap<ProjectId, String>);

impl Names {
    pub fn load(store: &Store) -> Result<Names> {
        Ok(Names(
            store
                .projects()?
                .into_iter()
                .map(|p| (p.id, p.name))
                .collect(),
        ))
    }

    pub fn name(&self, id: ProjectId) -> String {
        self.0.get(&id).cloned().unwrap_or_default()
    }

    pub fn view(&self, m: Memory) -> MemoryView {
        let project = self.name(m.project_id);
        view(m, project)
    }
}

/// The view of a memory. Its source tag is computed here, from the whole
/// provenance, before any token budget sheds part of it (decision B-220).
pub(crate) fn view(m: Memory, project: String) -> MemoryView {
    MemoryView {
        id: m.id,
        project,
        memory_type: m.memory_type,
        status: m.status,
        origin: m.origin,
        source_tag: Some(SourceTag::of(&m.provenance)),
        title: m.title,
        content: m.content,
        content_truncated: false,
        score: None,
        weak: false,
        rank: None,
        matched_by: None,
        importance: m.importance,
        tags: m.tags,
        anchors: m.anchors,
        provenance: m.provenance,
        supersedes_id: m.supersedes_id,
        created_at: m.created_at,
        updated_at: m.updated_at,
    }
}
