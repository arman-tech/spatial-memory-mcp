//! `link` (SPEC 8; SPEC 4, links): add, remove or list links in the current
//! project.
//!
//! Links made through the tool are `manual` and carry no commit: SPEC 4
//! has only machine-extracted links record the commit they were verified
//! against, and 3.0 extracts none (SPEC 12). `list` reports staleness with
//! `head_only`: a `verified_commit` equal to the project head (or a prefix
//! of it) is current, and anything else is left undecided rather than
//! called stale, because the daemon has no git ancestry walk.

use sm_core::error::Result;
use sm_core::protocol::{LinkAction, LinkParams, LinkReply};
use sm_core::types::{Link, LinkSource};
use sm_store::head_only;

use crate::budget::fit_whole;
use crate::dispatch::Ctx;

fn required<T: Clone>(v: &Option<T>, name: &str) -> Result<T> {
    v.clone().ok_or_else(|| {
        sm_core::error::CoreError::invalid_input(format!("this action needs {name}"))
    })
}

pub(crate) fn link(ctx: &Ctx<'_>, p: &LinkParams) -> Result<LinkReply> {
    let sh = ctx.sh;
    let project = ctx.project()?;
    let pid = project.project.id;
    let mut reply = LinkReply {
        action: p.action,
        link: None,
        removed: 0,
        links: Vec::new(),
        omitted: 0,
    };
    match p.action {
        LinkAction::Add => {
            let added = sh.store.add_link(
                pid,
                Link {
                    src: required(&p.src, "src")?,
                    dst: required(&p.dst, "dst")?,
                    relation: required(&p.relation, "relation")?,
                    source: LinkSource::Manual,
                    extractor: None,
                    verified_commit: None,
                    verified_at: None,
                },
            )?;
            reply.link = Some(added.link);
        }
        LinkAction::Remove => {
            reply.removed = sh.store.remove_link(
                pid,
                &required(&p.src, "src")?,
                &required(&p.dst, "dst")?,
                required(&p.relation, "relation")?,
            )?;
        }
        LinkAction::List => {
            let end = required(&p.end, "end")?;
            let head = project.identity.repo.as_ref().and_then(|r| r.head_commit());
            let links = sh.store.links_of(pid, &end, head_only(head.as_deref()))?;
            let fitted = fit_whole(links, ctx.budget(p.limit), 0);
            reply.links = fitted.items;
            reply.omitted = fitted.omitted;
        }
    }
    Ok(reply)
}
