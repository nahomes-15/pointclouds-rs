#![forbid(unsafe_code)]

pub mod correspondence;
pub mod icp;
pub mod icp_plane;

pub use correspondence::{find_correspondences, Correspondence};
pub use icp::{apply_transform, icp_point_to_point, IcpParams, IcpResult, RigidTransform};
pub use icp_plane::{icp_point_to_plane, IcpPlaneError};
