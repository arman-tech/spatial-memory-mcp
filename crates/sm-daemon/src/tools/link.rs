//! `link` (SPEC 8; SPEC 4, links): add, remove or list links in the current
//! project.
//!
//! Links made through the tool are `manual` and carry no commit: SPEC 4
//! has only machine-extracted links record the commit they were verified
//! against, and 3.0 extracts none (SPEC 12). `list` reports staleness with
//! `head_only`: a `verified_commit` equal to the project head (or a prefix
//! of it) is current, and anything else is left undecided rather than
//! called stale, because the daemon has no git ancestry walk.
//!
//! An automatic capture that waits for the judge (`pending`) is never
//! linked and never listed (B-181): `add` refuses one as either end, and
//! `list` refuses one as its end and leaves out a link to one (in the
//! store). `remove` is allowed; its reply is a count.

use sm_core::error::Result;
use sm_core::protocol::{LinkAction, LinkParams, LinkReply};
use sm_core::types::{Link, LinkEnd, LinkSource};
use sm_store::{head_only, pending_refused};

use crate::budget::{fit_whole, fixed_cost};
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
            let src = required(&p.src, "src")?;
            let dst = required(&p.dst, "dst")?;
            let relation = required(&p.relation, "relation")?;
            for end in [&src, &dst] {
                if let LinkEnd::Memory(id) = end
                    && sh
                        .store
                        .find_memory(id)?
                        .is_some_and(|m| !m.status.is_served())
                {
                    return Err(pending_refused(id, "linked"));
                }
            }
            let added = sh.store.add_link(
                pid,
                Link {
                    src,
                    dst,
                    relation,
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
            let budget = ctx.budget(p.limit);
            let skeleton = LinkReply {
                omitted: u32::MAX,
                ..reply.clone()
            };
            let fitted = fit_whole(links, budget, fixed_cost("link", &skeleton, budget)?);
            reply.links = fitted.items;
            reply.omitted = fitted.omitted;
        }
    }
    Ok(reply)
}
