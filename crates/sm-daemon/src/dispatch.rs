//! One request: the envelope check, the profile check, validation, then the
//! op (CONTRACTS 3.1).
//!
//! 1. `Envelope::peek` reads `v`, `build`, `id` and the op tag. A body that
//!    is not even that gets `invalid_input` with an empty id; the
//!    connection stays usable.
//! 2. Another `v` or `build` gets `version_mismatch`, except `shutdown`,
//!    which is accepted from any build for this profile.
//! 3. Unknown keys are refused: the request's own keys, and the arguments
//!    of the hook and command-line ops, whose structs do not deny unknown
//!    fields themselves (the tool params do). See e-report section 8.
//! 4. Another profile gets `invalid_input`.
//! 5. `Op::validate` and the absolute-path check of the file ops, then the
//!    op.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::ids::MemoryId;
use sm_core::protocol::{
    DaemonInfo, Envelope, Op, PROTOCOL_VERSION, Reply, Request, Response, decode_body,
};
use sm_core::types::Scope;

use crate::project::{self, read_scope};
use crate::state::{ProjectEntry, Shared, lock};
use crate::{admin, ambient, capture, tools};

/// What the connection does after writing the reply.
pub(crate) enum After {
    Nothing,
    Shutdown(String),
    /// Ambient recall: record what was injected, if still before the
    /// deadline (SPEC 6, step 6).
    RecordInjections {
        session: String,
        ids: Vec<MemoryId>,
        deadline: Instant,
    },
}

const REQUEST_KEYS: &[&str] = &[
    "v",
    "build",
    "id",
    "cwd",
    "session_id",
    "client",
    "profile",
    "deadline_ms",
    "call",
];

/// Argument keys of the ops whose params do not deny unknown fields.
fn allowed_args(op: &str) -> Option<&'static [&'static str]> {
    Some(match op {
        "shutdown" => &["reason"],
        "ambient_recall" => &["prompt", "top_k", "scope"],
        "capture" => &["event", "payload", "received_at"],
        "import" => &["path"],
        "export" => &["path", "project"],
        "migrate" => &["from", "default_project"],
        "reindex" => &[],
        "project_rename" => &["from", "to"],
        "project_delete" => &["name", "confirm"],
        "doctor_eval" => &["set_path", "baseline_path"],
        "doctor_numbers" => &[],
        _ => return None,
    })
}

fn unknown_keys(raw: &Value) -> Result<()> {
    let Some(obj) = raw.as_object() else {
        return Err(CoreError::invalid_input("a request must be a JSON object"));
    };
    if let Some(k) = obj.keys().find(|k| !REQUEST_KEYS.contains(&k.as_str())) {
        return Err(CoreError::invalid_input(format!(
            "unknown field `{k}` in the request"
        )));
    }
    let Some(call) = obj.get("call").and_then(Value::as_object) else {
        return Ok(());
    };
    if let Some(k) = call.keys().find(|k| *k != "op" && *k != "args") {
        return Err(CoreError::invalid_input(format!(
            "unknown field `{k}` in the call"
        )));
    }
    let op = call.get("op").and_then(Value::as_str).unwrap_or_default();
    if let (Some(allowed), Some(args)) = (
        allowed_args(op),
        call.get("args").and_then(Value::as_object),
    ) && let Some(k) = args.keys().find(|k| !allowed.contains(&k.as_str()))
    {
        return Err(CoreError::invalid_input(format!(
            "unknown field `{k}` in the arguments of `{op}`"
        )));
    }
    Ok(())
}

/// Context for one request.
pub(crate) struct Ctx<'a> {
    pub sh: &'a Arc<Shared>,
    pub req: &'a Request,
    pub deadline: Option<Instant>,
    /// The `spool/new/` file a drained capture was read from, claimed by
    /// the drain; `None` for a socket request.
    pub spool_file: Option<&'a Path>,
}

impl Ctx<'_> {
    /// `timeout` once the client's deadline has passed.
    pub fn check_deadline(&self, after: &str) -> Result<()> {
        match self.deadline {
            Some(d) if Instant::now() > d => Err(CoreError::new(
                ErrorCode::Timeout,
                format!(
                    "the request's deadline of {} ms passed after {after}",
                    self.req.deadline_ms.unwrap_or_default()
                ),
            )),
            _ => Ok(()),
        }
    }

    /// The reply budget in tokens: `limit`, else `output.token_budget`.
    pub fn budget(&self, limit: Option<u32>) -> usize {
        limit.unwrap_or(self.sh.settings.output.token_budget) as usize
    }

    pub fn project(&self) -> Result<ProjectEntry> {
        project::resolve(self.sh, &self.req.cwd)
    }

    pub fn scope(
        &self,
        scope: Option<Scope>,
    ) -> Result<(sm_core::traits::QueryScope, Option<ProjectEntry>)> {
        read_scope(self.sh, &self.req.cwd, scope)
    }

    pub fn session(&self) -> Option<&str> {
        self.req.session_id.as_deref().filter(|s| !s.is_empty())
    }
}

pub(crate) fn handle(sh: &Arc<Shared>, body: &[u8]) -> (Response, After) {
    let received = Instant::now();
    let env = match Envelope::peek(body) {
        Ok(e) => e,
        Err(e) => return (Response::err(String::new(), e), After::Nothing),
    };
    let raw: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return (
                Response::err(
                    env.id,
                    CoreError::invalid_input(format!("malformed message: {e}")),
                ),
                After::Nothing,
            );
        }
    };
    if env.check_version().is_err() {
        if !env.is_shutdown() {
            return (Response::version_mismatch(&env), After::Nothing);
        }
        // Shutdown is accepted from any build, for this profile only.
        let profile = raw.get("profile").and_then(Value::as_str);
        if profile != Some(sh.paths.profile.as_str()) {
            return (
                Response::err(env.id, wrong_profile(sh, profile)),
                After::Nothing,
            );
        }
        let reason = raw
            .pointer("/call/args/reason")
            .and_then(Value::as_str)
            .unwrap_or("version_mismatch")
            .to_owned();
        return (
            Response::ok(env.id, Reply::ShuttingDown),
            After::Shutdown(reason),
        );
    }
    if let Err(e) = unknown_keys(&raw) {
        return (Response::err(env.id, e), After::Nothing);
    }
    let req: Request = match decode_body(body) {
        Ok(r) => r,
        Err(e) => return (Response::err(env.id, e), After::Nothing),
    };
    handle_request(sh, &req, received, None)
}

/// Steps 4 and 5 for a request already decoded: a socket request, or a
/// capture read back from the spool file `spool_file` (checked when read).
pub(crate) fn handle_request(
    sh: &Arc<Shared>,
    req: &Request,
    received: Instant,
    spool_file: Option<&Path>,
) -> (Response, After) {
    if req.profile != sh.paths.profile {
        let e = wrong_profile(sh, Some(req.profile.as_str()));
        return (Response::err(req.id.clone(), e), After::Nothing);
    }
    if let Err(e) = req.call.validate().and_then(|()| absolute_paths(&req.call)) {
        return (Response::err(req.id.clone(), e), After::Nothing);
    }
    let ctx = Ctx {
        sh,
        req,
        deadline: req
            .deadline_ms
            .map(|ms| received + Duration::from_millis(ms)),
        spool_file,
    };
    let (result, after) = run_op(&ctx);
    let response = match result {
        Ok(reply) => Response::ok(req.id.clone(), reply),
        Err(e) => Response::err(req.id.clone(), e),
    };
    (response, after)
}

/// The file paths of `import`, `export`, `migrate` and `doctor_eval` are
/// absolute, resolved by the client (CONTRACTS 5, Requests): a
/// relative path would resolve against the daemon's own working folder
/// (e-report decision 36).
fn absolute_paths(call: &Op) -> Result<()> {
    let paths: Vec<(&str, &str)> = match call {
        Op::Import(p) => vec![("path", p.path.as_str())],
        Op::Export(p) => vec![("path", p.path.as_str())],
        Op::Migrate(p) => vec![("from", p.from.as_str())],
        Op::DoctorEval(p) => std::iter::once(("set_path", p.set_path.as_str()))
            .chain(p.baseline_path.as_deref().map(|b| ("baseline_path", b)))
            .collect(),
        _ => return Ok(()),
    };
    match paths.into_iter().find(|(_, p)| !Path::new(p).is_absolute()) {
        None => Ok(()),
        Some((field, p)) => Err(CoreError::invalid_input(format!(
            "`{}` needs an absolute path, not `{p}`",
            field
        ))
        .with_next_action(format!(
            "Resolve the path against your own working folder before sending `{}`; the \
             daemon's working folder is not yours.",
            call.name()
        ))),
    }
}

fn wrong_profile(sh: &Shared, got: Option<&str>) -> CoreError {
    CoreError::invalid_input(format!(
        "this daemon serves profile {}, not {}",
        sh.paths.profile,
        got.unwrap_or("(none)")
    ))
    .with_next_action("Connect to the endpoint of the request's own profile.")
}

fn none(r: Result<Reply>) -> (Result<Reply>, After) {
    (r, After::Nothing)
}

fn run_op(ctx: &Ctx<'_>) -> (Result<Reply>, After) {
    let sh = ctx.sh;
    match &ctx.req.call {
        Op::Hello => none(Ok(Reply::Hello(hello(sh)))),
        Op::Shutdown(p) => (Ok(Reply::ShuttingDown), After::Shutdown(p.reason.clone())),
        Op::Remember(p) => none(tools::remember::remember(ctx, p).map(Reply::Remember)),
        Op::Recall(p) => none(tools::recall::recall(ctx, p).map(Reply::Recall)),
        Op::Related(p) => none(tools::related::related(ctx, p).map(Reply::Related)),
        Op::Revise(p) => none(tools::edit::revise(ctx, p).map(Reply::Revise)),
        Op::Forget(p) => none(tools::edit::forget(ctx, p).map(Reply::Forget)),
        Op::Link(p) => none(tools::link::link(ctx, p).map(Reply::Link)),
        Op::Regions(p) => none(tools::maps::regions(ctx, p).map(Reply::Regions)),
        Op::Journey(p) => none(tools::maps::journey(ctx, p).map(Reply::Journey)),
        Op::Consolidate(p) => none(tools::consolidate::consolidate(ctx, p).map(Reply::Consolidate)),
        Op::Share(p) => none(admin::share(ctx, p).map(Reply::Share)),
        Op::Sync(p) => none(admin::sync(ctx, p).map(Reply::Sync)),
        Op::Status(p) => none(tools::status::status(ctx, p).map(|s| Reply::Status(Box::new(s)))),
        Op::Reinforce(p) => none(tools::recall::reinforce(ctx, p).map(Reply::Reinforce)),
        Op::AmbientRecall(p) => {
            let (reply, record) = ambient::ambient(ctx, p);
            let after = match record {
                Some((session, ids, deadline)) => After::RecordInjections {
                    session,
                    ids,
                    deadline,
                },
                None => After::Nothing,
            };
            (Ok(Reply::Ambient(reply)), after)
        }
        Op::Capture(p) => none(capture::enqueue(ctx, p).map(Reply::Captured)),
        Op::Import(p) => none(admin::import(ctx, p).map(Reply::Import)),
        Op::Export(p) => none(admin::export(ctx, p).map(Reply::Export)),
        Op::Migrate(p) => none(admin::migrate(ctx, p).map(Reply::Import)),
        Op::Reindex(_) => none(admin::reindex(ctx).map(Reply::Reindex)),
        Op::ProjectRename(p) => none(admin::project_rename(ctx, p).map(Reply::Project)),
        Op::ProjectDelete(p) => none(admin::project_delete(ctx, p).map(Reply::Project)),
        Op::DoctorEval(p) => none(admin::doctor_eval(ctx, p).map(|r| Reply::Eval(Box::new(r)))),
        Op::DoctorNumbers => none(admin::doctor_numbers(ctx).map(Reply::DoctorNumbers)),
    }
}

pub(crate) fn hello(sh: &Shared) -> DaemonInfo {
    let (state, state_error) = sh.state();
    DaemonInfo {
        protocol: PROTOCOL_VERSION,
        build: sm_core::BUILD.to_owned(),
        pid: std::process::id(),
        profile: sh.paths.profile.as_str().to_owned(),
        state,
        state_error,
        started_at: sh.started_at,
        ready_seconds: *lock(&sh.ready_seconds),
        model_id: sh.info.model_id.clone(),
        endpoint: sh.endpoint.display_path(),
    }
}

#[cfg(test)]
mod tests {
    use sm_core::protocol::{
        AmbientParams, CaptureParams, DoctorEvalParams, ExportParams, HookEvent, ImportParams,
        MigrateParams, ProjectDeleteParams, ProjectRenameParams, ReindexParams, ShutdownParams,
    };
    use sm_core::time::Timestamp;

    use super::*;

    /// Every op whose `sm-core` params do not deny unknown fields, with
    /// every field set, so each key it can send is serialised. The struct
    /// literals name every field: a field added in `sm-core` stops this
    /// compiling until it is added here, and then the test checks it.
    fn unchecked_ops() -> Vec<Op> {
        let s = || "/abs/x".to_owned();
        vec![
            Op::Shutdown(ShutdownParams { reason: s() }),
            Op::AmbientRecall(AmbientParams {
                prompt: s(),
                top_k: 3,
                scope: Scope::All,
            }),
            Op::Capture(CaptureParams {
                event: HookEvent::Stop,
                payload: serde_json::json!({ "session_id": "s" }),
                received_at: Timestamp::now(),
            }),
            Op::Import(ImportParams { path: s() }),
            Op::Export(ExportParams {
                path: s(),
                project: Some(s()),
            }),
            Op::Migrate(MigrateParams {
                from: s(),
                default_project: Some(s()),
            }),
            Op::Reindex(ReindexParams {}),
            Op::ProjectRename(ProjectRenameParams { from: s(), to: s() }),
            Op::ProjectDelete(ProjectDeleteParams {
                name: s(),
                confirm: true,
            }),
            Op::DoctorEval(DoctorEvalParams {
                set_path: s(),
                baseline_path: Some(s()),
            }),
            // B-125: no arguments, so none is let through.
            Op::DoctorNumbers,
        ]
    }

    /// B-102 drift guard: dispatch's `allowed_args` lists exactly the keys
    /// each unchecked `sm-core` parameter struct serialises, so a field
    /// added there is never refused as unknown, and no stale key is let
    /// through.
    #[test]
    fn s3_allowed_args_match_every_key_the_unchecked_params_serialise() {
        let ops = unchecked_ops();
        assert_eq!(
            ops.len(),
            11,
            "the ten ops of B-102 and doctor_numbers (B-125)"
        );
        for op in ops {
            let v = serde_json::to_value(&op).expect("encode an op");
            let name = v["op"].as_str().expect("an op tag").to_owned();
            assert_eq!(name, op.name());
            let allowed =
                allowed_args(&name).unwrap_or_else(|| panic!("`{name}` has no allowed_args list"));
            let mut sent: Vec<&str> = v
                .get("args")
                .and_then(Value::as_object)
                .map(|o| o.keys().map(String::as_str).collect())
                .unwrap_or_default();
            sent.sort_unstable();
            let mut listed = allowed.to_vec();
            listed.sort_unstable();
            assert_eq!(sent, listed, "the argument keys of `{name}`");
            // And the check lets the full arguments through.
            let raw = serde_json::json!({ "call": v });
            unknown_keys(&raw).unwrap_or_else(|e| panic!("`{name}`: {}", e.message));
        }
    }

    /// B-102 drift guard, the eleventh struct: `REQUEST_KEYS` lists exactly
    /// the keys a `Request` with every field set serialises. The literal
    /// names every field, so a field added in `sm-core` stops this
    /// compiling until it is added here.
    #[test]
    fn s3_request_keys_match_every_key_a_request_serialises() {
        let req = Request {
            v: PROTOCOL_VERSION,
            build: sm_core::BUILD.to_owned(),
            id: "r1".to_owned(),
            cwd: "/abs/x".to_owned(),
            session_id: Some("s".to_owned()),
            client: Some(sm_core::types::Client::Cursor),
            profile: sm_core::config::ProfileName::default_profile(),
            deadline_ms: Some(1),
            call: Op::Reindex(ReindexParams {}),
        };
        let v = serde_json::to_value(&req).expect("encode a request");
        let mut sent: Vec<&str> = v
            .as_object()
            .expect("a request is an object")
            .keys()
            .map(String::as_str)
            .collect();
        sent.sort_unstable();
        let mut listed = REQUEST_KEYS.to_vec();
        listed.sort_unstable();
        assert_eq!(sent, listed, "the keys of a request");
        unknown_keys(&v).unwrap_or_else(|e| panic!("a full request: {}", e.message));

        // B-125: `doctor_numbers` is a call with no arguments; any argument
        // sent with it is refused as unknown.
        let numbers = Request {
            call: Op::DoctorNumbers,
            ..req
        };
        let v = serde_json::to_value(&numbers).expect("encode a request");
        assert_eq!(v["call"], serde_json::json!({ "op": "doctor_numbers" }));
        unknown_keys(&v).unwrap_or_else(|e| panic!("doctor_numbers: {}", e.message));
        let mut extra = v.clone();
        extra["call"]["args"] = serde_json::json!({ "surprise": 1 });
        let e = unknown_keys(&extra).expect_err("an argument doctor_numbers does not take");
        assert!(e.message.contains("surprise"), "{}", e.message);
    }
}
