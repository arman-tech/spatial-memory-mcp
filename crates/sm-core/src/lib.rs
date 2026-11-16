//! Shared contracts for Spatial Memory 3.0.
//!
//! Every other crate depends on this one and on nothing else in the
//! workspace (SPEC section 3, Crates). It holds the domain types, the
//! configuration, the error codes, the traits behind every external engine,
//! the daemon message schema and the pure helpers several crates need to
//! agree on (hashing, timestamps, project identity, the metrics journal line
//! format and the ambient recall block). The contracts are frozen after
//! phase 0 (SPEC section 14); `docs/v3/CONTRACTS.md` describes them.

#![forbid(unsafe_code)]

/// Declares a closed set of string values: a `Copy` enum with `as_str`,
/// `ALL`, `Display`, `FromStr`, serde as a plain string and a JSON Schema
/// `enum` of the strings. Unknown strings are an `invalid_input` error.
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        pub enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident => $text:literal ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum $name {
            $( $(#[$vmeta])* $variant ),+
        }

        impl $name {
            /// Every value, in declaration order.
            pub const ALL: &'static [$name] = &[$($name::$variant),+];

            /// The value as written in JSON, TOML, SQL and front matter.
            pub const fn as_str(self) -> &'static str {
                match self {
                    $($name::$variant => $text),+
                }
            }
        }

        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::core::str::FromStr for $name {
            type Err = $crate::error::CoreError;

            fn from_str(s: &str) -> ::core::result::Result<Self, Self::Err> {
                match s {
                    $($text => Ok($name::$variant),)+
                    _ => Err($crate::error::CoreError::invalid_input(format!(
                        "unknown {} {:?}; expected one of: {}",
                        stringify!($name),
                        s,
                        [$($text),+].join(", ")
                    ))),
                }
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(&self, s: S) -> ::core::result::Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(d: D) -> ::core::result::Result<Self, D::Error> {
                let s = <::std::string::String as ::serde::Deserialize>::deserialize(d)?;
                s.parse().map_err(|e: $crate::error::CoreError| <D::Error as ::serde::de::Error>::custom(e.message))
            }
        }

        impl ::schemars::JsonSchema for $name {
            fn schema_name() -> ::std::borrow::Cow<'static, str> {
                ::std::borrow::Cow::Borrowed(stringify!($name))
            }

            fn json_schema(_: &mut ::schemars::SchemaGenerator) -> ::schemars::Schema {
                ::schemars::json_schema!({
                    "type": "string",
                    "enum": [$($text),+]
                })
            }
        }
    };
}

pub mod ambient;
pub mod config;
pub mod error;
pub mod eval;
pub mod hash;
pub mod ids;
pub mod journal;
pub mod paths;
pub mod project;
pub mod protocol;
pub mod spool;
pub mod time;
pub mod traits;
pub mod types;

#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub use error::{CoreError, ErrorCode, Result};

/// Dimension of every stored embedding (SPEC section 4: 768 little-endian
/// 32-bit floats per chunk). Both supported models produce 768 dimensions.
pub const EMBED_DIM: usize = 768;

/// The version of this build. Client and daemon are the same binary; a
/// client that finds a daemon reporting a different build asks it to exit and
/// starts a new one (SPEC section 3, rule 1, Transport).
pub const BUILD: &str = env!("CARGO_PKG_VERSION");
