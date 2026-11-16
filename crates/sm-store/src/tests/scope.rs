//! AT-12 (store part): project scope hides other projects; scope `all` sees
//! them; two profiles are two database files.

use sm_core::config::ProfileName;
use sm_core::paths::ProfilePaths;
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};

use super::*;

fn nearest_ids(store: &Store, text: &str, scope: QueryScope) -> Vec<MemoryId> {
    let q = fake_vector(text);
    store
        .nearest(&VectorQuery {
            vector: &q,
            scope,
            filter: MemoryFilter::default(),
            limit: 10,
        })
        .expect("vector search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

fn keyword_ids(store: &Store, text: &str, scope: QueryScope) -> Vec<MemoryId> {
    store
        .keyword_search(text, scope, &MemoryFilter::default(), 10)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

#[test]
fn at12_project_scope_hides_other_projects() {
    let fx = fixture();
    let a = project(&fx.store, "project-a");
    let b = project(&fx.store, "project-b");
    let text = "orders table uses soft delete";
    let in_a = add(&fx.store, a, text);
    let in_b = add(&fx.store, b, "kubernetes ingress certificate rotation");

    assert_eq!(
        nearest_ids(&fx.store, text, QueryScope::Project(b)),
        vec![in_b]
    );
    assert_eq!(
        nearest_ids(&fx.store, text, QueryScope::Project(a)),
        vec![in_a]
    );
    let all = nearest_ids(&fx.store, text, QueryScope::All);
    assert_eq!(all[0], in_a);
    assert!(all.contains(&in_b));

    assert!(keyword_ids(&fx.store, "soft delete", QueryScope::Project(b)).is_empty());
    assert_eq!(
        keyword_ids(&fx.store, "soft delete", QueryScope::Project(a)),
        vec![in_a]
    );
    assert_eq!(
        keyword_ids(&fx.store, "soft delete", QueryScope::All),
        vec![in_a]
    );

    let ids = |scope| -> Vec<MemoryId> {
        fx.store
            .memory_vectors(scope, &MemoryFilter::default())
            .expect("memory vectors")
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    };
    assert_eq!(ids(QueryScope::Project(b)), vec![in_b]);
    let mut both = vec![in_a, in_b];
    both.sort();
    assert_eq!(ids(QueryScope::All), both);

    // The anchor filter applies in each memory's own project.
    let mut anchored = note(a, "anchored in a");
    anchored.memory.anchors = vec!["table:orders".parse().expect("anchor")];
    let anchored = fx
        .store
        .insert_memory(anchored, Access::Local)
        .expect("insert")
        .id;
    assert_eq!(
        fx.store
            .memories_for_anchor(a, &"table:orders".parse().expect("anchor"))
            .expect("anchor memories"),
        vec![anchored]
    );
    assert!(
        fx.store
            .memories_for_anchor(b, &"table:orders".parse().expect("anchor"))
            .expect("anchor memories")
            .is_empty()
    );
}

#[test]
fn at12_two_profiles_are_two_database_files() {
    let root = tempfile::tempdir().expect("temporary folder");
    let paths = |name: &str| {
        ProfilePaths::new(
            root.path().to_path_buf(),
            ProfileName::parse(name).expect("profile name"),
        )
    };
    let work_paths = paths("work");
    let personal_paths = paths("personal");
    assert_ne!(work_paths.database(), personal_paths.database());

    let text = "the deploy key rotates monthly";
    let id = {
        let work = Store::open(&work_paths).expect("open work");
        work.configure_index(stamp()).expect("configure");
        let personal = Store::open(&personal_paths).expect("open personal");
        personal.configure_index(stamp()).expect("configure");
        assert_eq!(work.path(), work_paths.database());
        assert_eq!(personal.path(), personal_paths.database());

        // The same project identity in both profiles.
        let pw = project(&work, "shared-name");
        let pp = project(&personal, "shared-name");
        let id = add(&work, pw, text);

        assert_eq!(personal.memory_count().expect("count"), 0);
        assert_eq!(personal.find_memory(&id).expect("find"), None);
        assert!(nearest_ids(&personal, text, QueryScope::All).is_empty());
        assert!(nearest_ids(&personal, text, QueryScope::Project(pp)).is_empty());
        assert!(keyword_ids(&personal, "deploy key", QueryScope::All).is_empty());
        assert_eq!(nearest_ids(&work, text, QueryScope::All), vec![id]);
        id
    };

    // Reopened, each profile still holds only its own data.
    let work = Store::open(&work_paths).expect("reopen work");
    work.configure_index(stamp()).expect("configure");
    let personal = Store::open(&personal_paths).expect("reopen personal");
    personal.configure_index(stamp()).expect("configure");
    assert_eq!(work.memory_count().expect("count"), 1);
    assert_eq!(work.get_memory(&id).expect("get").content, text);
    assert_eq!(personal.memory_count().expect("count"), 0);
    assert_code(personal.get_memory(&id), ErrorCode::NotFound);
}
