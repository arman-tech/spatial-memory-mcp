//! `revise` and `forget` (SPEC 8; B-30, B-72).
//!
//! `revise` edits fields in place, or, with `supersede_with`, stores a new
//! memory through the `remember` path that supersedes the old one in the
//! same transaction. New content is redacted and re-chunked; its chunks
//! wait for the bulk queue like any new memory's, so a content change
//! needs the model's tokenizer and is refused while the model loads.
//!
//! **A new title re-embeds** (stream H 15). The document the bulk queue
//! embeds is the title with the chunk's text, so a chunk embedded before a
//! title change holds the old title. A revise that changes the title
//! rebuilds the memory's chunks from its stored content, in the same
//! transaction as the title, and they wait for the bulk queue; like a
//! content change it needs the tokenizer, is refused while the model
//! loads, and says `reembedded`. A captured memory is chunked again with
//! the capture index cap it was stored with (`capture.max_index_tokens`),
//! so what is indexed stays the same. A title set to the one the memory
//! has changes nothing and needs no model. Every `revise` holds
//! `Shared::revising`, so no other revision changes the content between
//! the read and the rebuild.
//!
//! `forget` deletes by identifier or by filter. More than one memory
//! without `confirm: true` deletes nothing and fails with
//! `confirm_required`, saying how many would go (B-30; the contract's
//! "more than one memory" is followed literally, so one memory needs no
//! confirmation). A filter deletes only local memories: shared ones change
//! only through `sync` (B-72). Deleting by identifier is all or nothing;
//! a shared memory among the ids fails the call with `shared_read_only`.

use std::collections::HashSet;

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::ids::MemoryId;
use sm_core::protocol::RememberOutcome;
use sm_core::protocol::{ForgetFilter, ForgetParams, ForgetReply, ReviseParams, ReviseReply};
use sm_core::traits::SecretScanner;
use sm_core::types::{Origin, ProvenanceSource, Scope};
use sm_store::{Access, ContentChange, MemoryListFilter, Revision};

use crate::dispatch::Ctx;
use crate::state::lock;
use crate::tools::remember::{pending_chunks, prepare_item, store};
use crate::view::Names;

pub(crate) fn revise(ctx: &Ctx<'_>, p: &ReviseParams) -> Result<ReviseReply> {
    let sh = ctx.sh;
    let _revising = lock(&sh.revising);
    let old = sh.store.get_memory(&p.id)?;
    if let Some(item) = &p.supersede_with {
        if let Some(other) = item.supersedes
            && other != p.id
        {
            return Err(CoreError::invalid_input(format!(
                "supersede_with.supersedes is {other}, but the memory revised is {}",
                p.id
            )));
        }
        let ready = sh.ready()?;
        let project = ctx.project()?;
        let mut item = item.clone();
        item.supersedes = Some(p.id);
        let (mut input, _) = prepare_item(ctx, &ready, &project, &item)?;
        if input.memory.project_id != old.project_id {
            // The new memory belongs where the old one is; the current
            // repository's commit says nothing about that project.
            input.memory.project_id = old.project_id;
            input.memory.provenance.commit_sha = None;
        }
        let names = Names::load(&sh.store)?;
        let name = names.name(old.project_id);
        let inserted = store(sh, &ctx.req.id, &name, vec![input])?;
        let Some(new) = inserted.into_iter().next() else {
            return Err(CoreError::internal("the store returned no memory"));
        };
        let memory = sh.store.get_memory(&new.id)?;
        return Ok(ReviseReply {
            memory: names.view(memory),
            superseded: new.superseded,
            reembedded: new.outcome == RememberOutcome::Created,
        });
    }

    let title = p.title.as_ref().map(|t| sh.scanner.redact(t).text);
    let content = match (&p.content, &title) {
        (Some(text), _) => {
            let ready = sh.ready()?;
            let redacted = sh.scanner.redact(text);
            let (chunks, _) = pending_chunks(sh, &ready, &redacted.text, None)?;
            Some(ContentChange {
                content: redacted.text,
                chunks,
            })
        }
        (None, Some(t)) if old.title.as_deref() != Some(t.as_str()) => {
            let ready = sh.ready()?;
            let cap = (old.provenance.source == ProvenanceSource::Capture)
                .then_some(sh.settings.capture.max_index_tokens as usize);
            let (chunks, _) = pending_chunks(sh, &ready, &old.content, cap)?;
            Some(ContentChange {
                content: old.content.clone(),
                chunks,
            })
        }
        (None, _) => None,
    };
    let reembedded = content.is_some();
    let rev = Revision {
        content,
        title,
        memory_type: p.memory_type,
        status: p.status,
        tags: p.tags.clone(),
        anchors: p.anchors.clone(),
        importance: p.importance,
    };
    let memory = sh.store.revise_memory(&p.id, rev, Access::Local)?;
    if reembedded {
        sh.after_write();
    } else {
        sh.drop_maps();
    }
    let names = Names::load(&sh.store)?;
    Ok(ReviseReply {
        memory: names.view(memory),
        superseded: None,
        reembedded,
    })
}

fn confirm_required(n: usize) -> CoreError {
    CoreError::new(
        ErrorCode::ConfirmRequired,
        format!("{n} memories would be deleted; nothing was deleted"),
    )
    .with_next_action("Repeat the call with confirm: true to delete them.")
}

/// The local memories a filter matches, in id order.
fn matching(ctx: &Ctx<'_>, f: &ForgetFilter) -> Result<Vec<MemoryId>> {
    let project = match f.scope.unwrap_or_default() {
        Scope::Project => Some(ctx.project()?.project.id),
        Scope::All => None,
    };
    let list = ctx.sh.store.list_memories(&MemoryListFilter {
        project,
        types: f.memory_type.clone(),
        statuses: f.status.clone(),
        tags: f.tag.iter().cloned().collect(),
        anchor: f.anchor.clone(),
        origin: Some(Origin::Local),
        after: None,
        limit: 0,
    })?;
    Ok(list
        .into_iter()
        .filter(|m| f.created_before.is_none_or(|t| m.created_at < t))
        .map(|m| m.id)
        .collect())
}

pub(crate) fn forget(ctx: &Ctx<'_>, p: &ForgetParams) -> Result<ForgetReply> {
    let sh = ctx.sh;
    let ids: Vec<MemoryId> = match &p.filter {
        Some(f) => matching(ctx, f)?,
        None => {
            let mut seen = HashSet::new();
            p.ids.iter().copied().filter(|i| seen.insert(*i)).collect()
        }
    };
    if ids.len() > 1 && !p.confirm {
        return Err(confirm_required(ids.len()));
    }
    if ids.is_empty() {
        return Ok(ForgetReply {
            deleted: Vec::new(),
        });
    }
    sh.store.forget(&ids, Access::Local)?;
    sh.drop_maps();
    Ok(ForgetReply { deleted: ids })
}
