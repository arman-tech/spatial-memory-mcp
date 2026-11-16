//! The spatial map (SPEC section 7).
//!
//! A pure library: no storage, no embedding model and no I/O. The daemon
//! (stream E) reads chunk vectors from the store, calls [`regions`] and
//! [`coordinates`], and caches the run under [`RegionRun::input_hash`].
//!
//! - [`HolomapProjector`] implements [`sm_core::traits::Projector`] with
//!   `holomap` 0.3.0 (cosine metric, spectral initialisation, fixed seed).
//! - [`HdbscanClusterer`] implements [`sm_core::traits::Clusterer`] with
//!   `hdbscan` 0.12.0 (Euclidean on the projected points, excess-of-mass
//!   selection, `max_cluster_size`).
//! - [`regions`] is the per-project pipeline: one vector per memory, no
//!   clustering under 30 memories, a deterministic 2,000-memory sample above
//!   the projection cap with nearest-centre assignment for the rest, and
//!   each region's members ranked by [`CENTRALITY`].
//! - [`journey`] walks the great circle between two memories and returns the
//!   nearest memory not already on the path at each step.
//!
//! Every function is deterministic: the same input gives bit-identical
//! output on every run and every thread count (AT-6). Both libraries are
//! serial, and nothing here reads a clock, a random source or a hash map's
//! iteration order.

#![forbid(unsafe_code)]

pub mod clusterer;
pub mod journey;
pub mod pipeline;
pub mod projector;
pub mod quality;
pub mod vector;

pub use clusterer::HdbscanClusterer;
pub use journey::{Endpoint, JourneyRequest, PathStep, journey, slerp_path};
pub use pipeline::{
    ALGORITHM, CENTRALITY, MapParams, MemoryVectors, RankedMember, RegionResult, RegionRun,
    coordinates, input_hash, regions, sample_order,
};
pub use projector::HolomapProjector;
pub use quality::adjusted_rand_index;
pub use vector::{cosine, memory_vector};
