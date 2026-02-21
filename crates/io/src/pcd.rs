use pointclouds_core::PointCloud;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

/// Reads a PCD file (ASCII or binary format).
pub fn read_pcd(path: impl AsRef<Path>) -> io::Result<PointCloud> {
    let raw = fs::read(path)?;

    // Find the DATA line to determine format
    let header_str = find_header(&raw)?;
    let data_format = parse_data_format(&header_str)?;
    let num_points = parse_points_count(&header_str)?;
    let field_names = parse_fields(&header_str);

    match data_format {
        DataFormat::Ascii => read_pcd_ascii(&raw, &field_names),
        DataFormat::Binary => read_pcd_binary(&raw, num_points, &field_names),
    }
}

/// Writes a PCD file in ASCII format.
pub fn write_pcd(path: impl AsRef<Path>, cloud: &PointCloud) -> io::Result<()> {
    let mut out = String::new();
    out.push_str("# .PCD v0.7 - Point Cloud Data file format\n");
    out.push_str("VERSION 0.7\n");
    out.push_str("FIELDS x y z\n");
    out.push_str("SIZE 4 4 4\n");
    out.push_str("TYPE F F F\n");
    out.push_str("COUNT 1 1 1\n");
    out.push_str(&format!("WIDTH {}\n", cloud.len()));
    out.push_str("HEIGHT 1\n");
    out.push_str("VIEWPOINT 0 0 0 1 0 0 0\n");
    out.push_str(&format!("POINTS {}\n", cloud.len()));
    out.push_str("DATA ascii\n");

    for i in 0..cloud.len() {
        out.push_str(&format!("{} {} {}\n", cloud.x[i], cloud.y[i], cloud.z[i]));
    }

    fs::write(path, out)
}

/// Writes a PCD file in binary format.
pub fn write_pcd_binary(path: impl AsRef<Path>, cloud: &PointCloud) -> io::Result<()> {
    let mut header = String::new();
    header.push_str("# .PCD v0.7 - Point Cloud Data file format\n");
    header.push_str("VERSION 0.7\n");
    header.push_str("FIELDS x y z\n");
    header.push_str("SIZE 4 4 4\n");
    header.push_str("TYPE F F F\n");
    header.push_str("COUNT 1 1 1\n");
    header.push_str(&format!("WIDTH {}\n", cloud.len()));
    header.push_str("HEIGHT 1\n");
    header.push_str("VIEWPOINT 0 0 0 1 0 0 0\n");
    header.push_str(&format!("POINTS {}\n", cloud.len()));
    header.push_str("DATA binary\n");

    let header_bytes = header.as_bytes();
    let point_size = 3 * 4; // 3 floats * 4 bytes each
    let mut buf = Vec::with_capacity(header_bytes.len() + cloud.len() * point_size);
    buf.extend_from_slice(header_bytes);

    for i in 0..cloud.len() {
        buf.extend_from_slice(&cloud.x[i].to_le_bytes());
        buf.extend_from_slice(&cloud.y[i].to_le_bytes());
        buf.extend_from_slice(&cloud.z[i].to_le_bytes());
    }

    fs::write(path, buf)
}

// --- Internal helpers ---

#[derive(Debug, PartialEq)]
enum DataFormat {
    Ascii,
    Binary,
}

/// Extracts the header portion as a UTF-8 string (everything up to and including the DATA line).
fn find_header(raw: &[u8]) -> io::Result<String> {
    // Scan for "\nDATA " or file starting with "DATA " (unlikely but handled)
    // The header ends at the newline after the DATA line.
    let text = std::str::from_utf8(raw)
        .ok()
        .or_else(|| {
            // For binary files, the header is ASCII but the body is binary.
            // Find the end of the DATA line by scanning for "DATA" followed by a newline.
            find_data_line_end(raw).and_then(|end| std::str::from_utf8(&raw[..end]).ok())
        })
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "PCD header is not valid UTF-8")
        })?;

    // Find the DATA line
    for line in text.lines() {
        if line.trim_start().starts_with("DATA") {
            // Return everything up to and including this line
            let offset = text
                .find(line)
                .map(|pos| pos + line.len())
                .unwrap_or(text.len());
            return Ok(text[..offset].to_string());
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "PCD file missing DATA line",
    ))
}

/// Finds the byte offset just past the newline ending the DATA line.
fn find_data_line_end(raw: &[u8]) -> Option<usize> {
    let data_marker = b"DATA";
    for i in 0..raw.len().saturating_sub(data_marker.len()) {
        if (i == 0 || raw[i - 1] == b'\n') && raw[i..].starts_with(data_marker) {
            // Find the newline after this
            if let Some(offset) = raw[i..].iter().position(|&b| b == b'\n') {
                return Some(i + offset + 1);
            }
            return Some(raw.len());
        }
    }
    None
}

fn parse_data_format(header: &str) -> io::Result<DataFormat> {
    for line in header.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("DATA") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                return match parts[1] {
                    "ascii" => Ok(DataFormat::Ascii),
                    "binary" => Ok(DataFormat::Binary),
                    other => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        format!("unsupported PCD DATA format: {}", other),
                    )),
                };
            }
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "PCD file missing DATA line",
    ))
}

fn parse_points_count(header: &str) -> io::Result<usize> {
    for line in header.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("POINTS") || trimmed.starts_with("WIDTH") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 && trimmed.starts_with("POINTS") {
                return parts[1].parse::<usize>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid POINTS value: {}", e),
                    )
                });
            }
        }
    }

    // Fall back to WIDTH if POINTS is not found
    for line in header.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("WIDTH") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 2 {
                return parts[1].parse::<usize>().map_err(|e| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("invalid WIDTH value: {}", e),
                    )
                });
            }
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "PCD file missing POINTS/WIDTH header",
    ))
}

fn parse_fields(header: &str) -> Vec<String> {
    for line in header.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("FIELDS") {
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            return parts[1..].iter().map(|s| s.to_string()).collect();
        }
    }
    // Default to x y z if no FIELDS line found
    vec!["x".to_string(), "y".to_string(), "z".to_string()]
}

fn read_pcd_ascii(raw: &[u8], _field_names: &[String]) -> io::Result<PointCloud> {
    let content = std::str::from_utf8(raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {}", e)))?;

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();

    let mut in_data = false;
    for line in content.lines() {
        if line.trim_start().starts_with("DATA") {
            in_data = true;
            continue;
        }
        if !in_data || line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }

        let px = parts[0].parse::<f32>().unwrap_or(0.0);
        let py = parts[1].parse::<f32>().unwrap_or(0.0);
        let pz = parts[2].parse::<f32>().unwrap_or(0.0);
        x.push(px);
        y.push(py);
        z.push(pz);
    }

    Ok(PointCloud::from_xyz(x, y, z))
}

fn read_pcd_binary(
    raw: &[u8],
    num_points: usize,
    field_names: &[String],
) -> io::Result<PointCloud> {
    // Find the byte offset where binary data starts (right after "DATA binary\n")
    let data_offset = find_data_line_end(raw).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "cannot find DATA line in binary PCD",
        )
    })?;

    let num_fields = field_names.len();
    let point_byte_size = num_fields * 4; // Each field is an f32 (4 bytes)
    let data_slice = &raw[data_offset..];
    let expected_size = num_points * point_byte_size;

    if data_slice.len() < expected_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "binary PCD data too short: have {} bytes, expected {} ({} points x {} fields x 4)",
                data_slice.len(),
                expected_size,
                num_points,
                num_fields
            ),
        ));
    }

    // Find indices of x, y, z fields
    let idx_x = field_names.iter().position(|n| n == "x");
    let idx_y = field_names.iter().position(|n| n == "y");
    let idx_z = field_names.iter().position(|n| n == "z");

    let (idx_x, idx_y, idx_z) = match (idx_x, idx_y, idx_z) {
        (Some(ix), Some(iy), Some(iz)) => (ix, iy, iz),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "binary PCD file missing x, y, z fields",
            ));
        }
    };

    let mut x = Vec::with_capacity(num_points);
    let mut y = Vec::with_capacity(num_points);
    let mut z = Vec::with_capacity(num_points);

    let mut cursor = io::Cursor::new(data_slice);
    let mut point_buf = vec![0u8; point_byte_size];

    for _ in 0..num_points {
        cursor.read_exact(&mut point_buf)?;

        let read_f32_at = |field_idx: usize| -> f32 {
            let byte_offset = field_idx * 4;
            let bytes = [
                point_buf[byte_offset],
                point_buf[byte_offset + 1],
                point_buf[byte_offset + 2],
                point_buf[byte_offset + 3],
            ];
            f32::from_le_bytes(bytes)
        };

        x.push(read_f32_at(idx_x));
        y.push(read_f32_at(idx_y));
        z.push(read_f32_at(idx_z));
    }

    Ok(PointCloud::from_xyz(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::NamedTempFile;

    #[test]
    fn pcd_roundtrip() {
        let cloud = PointCloud::from_xyz(
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        );
        let tmp = NamedTempFile::new().unwrap();
        write_pcd(tmp.path(), &cloud).unwrap();
        let loaded = read_pcd(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.x, cloud.x);
        assert_eq!(loaded.y, cloud.y);
        assert_eq!(loaded.z, cloud.z);
    }

    #[test]
    fn pcd_empty_cloud() {
        let cloud = PointCloud::new();
        let tmp = NamedTempFile::new().unwrap();
        write_pcd(tmp.path(), &cloud).unwrap();
        let loaded = read_pcd(tmp.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn pcd_read_sample() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/bunny.pcd");
        let cloud = read_pcd(path).unwrap();
        // The sample bunny.pcd has 1 point at (0, 0, 0)
        assert_eq!(cloud.len(), 1);
        assert_eq!(cloud.x[0], 0.0);
        assert_eq!(cloud.y[0], 0.0);
        assert_eq!(cloud.z[0], 0.0);
    }

    #[test]
    fn pcd_binary_roundtrip() {
        let cloud = PointCloud::from_xyz(
            vec![1.5, -2.5, 3.0],
            vec![4.0, 5.25, -6.0],
            vec![7.0, 8.0, 9.125],
        );
        let tmp = NamedTempFile::new().unwrap();
        write_pcd_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_pcd(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.x, cloud.x);
        assert_eq!(loaded.y, cloud.y);
        assert_eq!(loaded.z, cloud.z);
    }

    #[test]
    fn pcd_binary_empty() {
        let cloud = PointCloud::new();
        let tmp = NamedTempFile::new().unwrap();
        write_pcd_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_pcd(tmp.path()).unwrap();
        assert!(loaded.is_empty());
    }

    proptest! {
        #[test]
        fn pcd_roundtrip_preserves_data(
            pts in prop::collection::vec(
                (-1000.0f32..1000.0f32, -1000.0f32..1000.0f32, -1000.0f32..1000.0f32),
                0..200
            )
        ) {
            let x: Vec<f32> = pts.iter().map(|p| p.0).collect();
            let y: Vec<f32> = pts.iter().map(|p| p.1).collect();
            let z: Vec<f32> = pts.iter().map(|p| p.2).collect();
            let cloud = PointCloud::from_xyz(x, y, z);

            // Test ASCII roundtrip
            let tmp = NamedTempFile::new().unwrap();
            write_pcd(tmp.path(), &cloud).unwrap();
            let loaded = read_pcd(tmp.path()).unwrap();

            prop_assert_eq!(loaded.len(), cloud.len());
            for i in 0..cloud.len() {
                prop_assert_eq!(loaded.x[i], cloud.x[i]);
                prop_assert_eq!(loaded.y[i], cloud.y[i]);
                prop_assert_eq!(loaded.z[i], cloud.z[i]);
            }
        }

        #[test]
        fn pcd_binary_roundtrip_preserves_data(
            pts in prop::collection::vec(
                (-1000.0f32..1000.0f32, -1000.0f32..1000.0f32, -1000.0f32..1000.0f32),
                0..200
            )
        ) {
            let x: Vec<f32> = pts.iter().map(|p| p.0).collect();
            let y: Vec<f32> = pts.iter().map(|p| p.1).collect();
            let z: Vec<f32> = pts.iter().map(|p| p.2).collect();
            let cloud = PointCloud::from_xyz(x, y, z);

            let tmp = NamedTempFile::new().unwrap();
            write_pcd_binary(tmp.path(), &cloud).unwrap();
            let loaded = read_pcd(tmp.path()).unwrap();

            prop_assert_eq!(loaded.len(), cloud.len());
            for i in 0..cloud.len() {
                prop_assert_eq!(loaded.x[i], cloud.x[i]);
                prop_assert_eq!(loaded.y[i], cloud.y[i]);
                prop_assert_eq!(loaded.z[i], cloud.z[i]);
            }
        }
    }
}
