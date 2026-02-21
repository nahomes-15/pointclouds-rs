use pointclouds_core::{Colors, Normals, PointCloud};
use std::fs;
use std::io::{self, BufWriter, Write as _};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
}

/// Property type as declared in the PLY header.
#[derive(Debug, Clone, Copy)]
enum PropType {
    Float,
    Uchar,
}

impl PropType {
    fn byte_size(self) -> usize {
        match self {
            PropType::Float => 4,
            PropType::Uchar => 1,
        }
    }
}

/// Parsed header information.
struct PlyHeader {
    format: PlyFormat,
    vertex_count: usize,
    property_names: Vec<String>,
    property_types: Vec<PropType>,
    header_end_offset: usize, // byte offset just after "end_header\n"
}

fn parse_ply_header(data: &[u8]) -> io::Result<PlyHeader> {
    // Find end_header line
    let header_str = std::str::from_utf8(data).map_err(|_| {
        // Might be binary with non-UTF-8 bytes after header; find end_header manually
        io::Error::new(io::ErrorKind::InvalidData, "PLY header not valid UTF-8")
    });

    // Try to find "end_header\n" in raw bytes
    let end_marker = b"end_header\n";
    let header_end = find_bytes(data, end_marker).ok_or_else(|| {
        // Try with \r\n
        io::Error::new(io::ErrorKind::InvalidData, "missing end_header in PLY file")
    })?;
    let header_end_offset = header_end + end_marker.len();

    let header_bytes = &data[..header_end];
    let header_text = std::str::from_utf8(header_bytes)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "PLY header not valid UTF-8"))?;

    let _ = header_str; // suppress unused warning

    let mut format = None;
    let mut vertex_count: usize = 0;
    let mut property_names: Vec<String> = Vec::new();
    let mut property_types: Vec<PropType> = Vec::new();
    let mut in_vertex_element = false;
    let mut seen_ply_magic = false;

    for line in header_text.lines() {
        let line = line.trim();

        if !seen_ply_magic {
            if line == "ply" {
                seen_ply_magic = true;
                continue;
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "file does not start with 'ply'",
                ));
            }
        }

        if line.starts_with("format") {
            if line.contains("ascii") {
                format = Some(PlyFormat::Ascii);
            } else if line.contains("binary_little_endian") {
                format = Some(PlyFormat::BinaryLittleEndian);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("unsupported PLY format: {}", line),
                ));
            }
        } else if line.starts_with("element vertex") {
            in_vertex_element = true;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 3 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "invalid element vertex line",
                ));
            }
            vertex_count = parts[2].parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid vertex count: {}", e),
                )
            })?;
        } else if line.starts_with("element") {
            in_vertex_element = false;
        } else if line.starts_with("property") && in_vertex_element {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let ptype = match parts[1] {
                    "float" | "float32" => PropType::Float,
                    "uchar" | "uint8" => PropType::Uchar,
                    "double" | "float64" => PropType::Float, // treat as float for reading
                    other => {
                        return Err(io::Error::new(
                            io::ErrorKind::Unsupported,
                            format!("unsupported property type: {}", other),
                        ));
                    }
                };
                property_types.push(ptype);
                property_names.push(parts[2].to_string());
            }
        }
    }

    let format = format
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "PLY format line missing"))?;

    Ok(PlyHeader {
        format,
        vertex_count,
        property_names,
        property_types,
        header_end_offset,
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

pub fn read_ply(path: impl AsRef<Path>) -> io::Result<PointCloud> {
    let data = fs::read(&path)?;
    let header = parse_ply_header(&data)?;

    // Find column indices
    let idx_x = header.property_names.iter().position(|n| n == "x");
    let idx_y = header.property_names.iter().position(|n| n == "y");
    let idx_z = header.property_names.iter().position(|n| n == "z");

    let (idx_x, idx_y, idx_z) = match (idx_x, idx_y, idx_z) {
        (Some(ix), Some(iy), Some(iz)) => (ix, iy, iz),
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PLY file missing required x, y, z properties",
            ));
        }
    };

    let idx_nx = header.property_names.iter().position(|n| n == "nx");
    let idx_ny = header.property_names.iter().position(|n| n == "ny");
    let idx_nz = header.property_names.iter().position(|n| n == "nz");

    let idx_red = header.property_names.iter().position(|n| n == "red");
    let idx_green = header.property_names.iter().position(|n| n == "green");
    let idx_blue = header.property_names.iter().position(|n| n == "blue");

    let has_normals = idx_nx.is_some() && idx_ny.is_some() && idx_nz.is_some();
    let has_colors = idx_red.is_some() && idx_green.is_some() && idx_blue.is_some();

    let vertex_count = header.vertex_count;

    let mut x = Vec::with_capacity(vertex_count);
    let mut y = Vec::with_capacity(vertex_count);
    let mut z = Vec::with_capacity(vertex_count);
    let mut nx_vec = Vec::with_capacity(if has_normals { vertex_count } else { 0 });
    let mut ny_vec = Vec::with_capacity(if has_normals { vertex_count } else { 0 });
    let mut nz_vec = Vec::with_capacity(if has_normals { vertex_count } else { 0 });
    let mut r_vec = Vec::with_capacity(if has_colors { vertex_count } else { 0 });
    let mut g_vec = Vec::with_capacity(if has_colors { vertex_count } else { 0 });
    let mut b_vec = Vec::with_capacity(if has_colors { vertex_count } else { 0 });

    match header.format {
        PlyFormat::Ascii => {
            let body = std::str::from_utf8(&data[header.header_end_offset..]).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidData, "PLY body not valid UTF-8")
            })?;
            let mut count = 0usize;
            for line in body.lines() {
                if count >= vertex_count {
                    break;
                }
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < header.property_names.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "vertex line has {} fields, expected {}",
                            parts.len(),
                            header.property_names.len()
                        ),
                    ));
                }

                let parse_f32 = |idx: usize| -> io::Result<f32> {
                    parts[idx].parse::<f32>().map_err(|e| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("failed to parse float: {}", e),
                        )
                    })
                };

                x.push(parse_f32(idx_x)?);
                y.push(parse_f32(idx_y)?);
                z.push(parse_f32(idx_z)?);

                if has_normals {
                    nx_vec.push(parse_f32(idx_nx.unwrap())?);
                    ny_vec.push(parse_f32(idx_ny.unwrap())?);
                    nz_vec.push(parse_f32(idx_nz.unwrap())?);
                }

                if has_colors {
                    let parse_u8 = |idx: usize| -> io::Result<u8> {
                        parts[idx].parse::<u8>().map_err(|e| {
                            io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!("failed to parse color byte: {}", e),
                            )
                        })
                    };
                    r_vec.push(parse_u8(idx_red.unwrap())?);
                    g_vec.push(parse_u8(idx_green.unwrap())?);
                    b_vec.push(parse_u8(idx_blue.unwrap())?);
                }

                count += 1;
            }
        }
        PlyFormat::BinaryLittleEndian => {
            let body = &data[header.header_end_offset..];
            let stride: usize = header.property_types.iter().map(|t| t.byte_size()).sum();
            let needed = vertex_count * stride;
            if body.len() < needed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "PLY binary body too short: need {} bytes, got {}",
                        needed,
                        body.len()
                    ),
                ));
            }

            for vi in 0..vertex_count {
                let row = &body[vi * stride..];
                // Compute byte offset for each property
                let read_f32_at = |prop_idx: usize| -> f32 {
                    let off: usize = header.property_types[..prop_idx]
                        .iter()
                        .map(|t| t.byte_size())
                        .sum();
                    f32::from_le_bytes([row[off], row[off + 1], row[off + 2], row[off + 3]])
                };
                let read_u8_at = |prop_idx: usize| -> u8 {
                    let off: usize = header.property_types[..prop_idx]
                        .iter()
                        .map(|t| t.byte_size())
                        .sum();
                    row[off]
                };

                x.push(read_f32_at(idx_x));
                y.push(read_f32_at(idx_y));
                z.push(read_f32_at(idx_z));

                if has_normals {
                    nx_vec.push(read_f32_at(idx_nx.unwrap()));
                    ny_vec.push(read_f32_at(idx_ny.unwrap()));
                    nz_vec.push(read_f32_at(idx_nz.unwrap()));
                }

                if has_colors {
                    r_vec.push(read_u8_at(idx_red.unwrap()));
                    g_vec.push(read_u8_at(idx_green.unwrap()));
                    b_vec.push(read_u8_at(idx_blue.unwrap()));
                }
            }
        }
    }

    let mut cloud = PointCloud::from_xyz(x, y, z);

    if has_normals {
        cloud.normals = Some(Normals {
            nx: nx_vec,
            ny: ny_vec,
            nz: nz_vec,
        });
    }

    if has_colors {
        cloud.colors = Some(Colors {
            r: r_vec,
            g: g_vec,
            b: b_vec,
        });
    }

    Ok(cloud)
}

/// Write a PLY file in ASCII format.
pub fn write_ply(path: impl AsRef<Path>, cloud: &PointCloud) -> io::Result<()> {
    let mut out = String::new();

    out.push_str("ply\n");
    out.push_str("format ascii 1.0\n");
    out.push_str(&format!("element vertex {}\n", cloud.len()));
    out.push_str("property float x\n");
    out.push_str("property float y\n");
    out.push_str("property float z\n");

    if cloud.normals.is_some() {
        out.push_str("property float nx\n");
        out.push_str("property float ny\n");
        out.push_str("property float nz\n");
    }

    if cloud.colors.is_some() {
        out.push_str("property uchar red\n");
        out.push_str("property uchar green\n");
        out.push_str("property uchar blue\n");
    }

    out.push_str("end_header\n");

    for i in 0..cloud.len() {
        out.push_str(&format!("{} {} {}", cloud.x[i], cloud.y[i], cloud.z[i]));

        if let Some(ref normals) = cloud.normals {
            out.push_str(&format!(
                " {} {} {}",
                normals.nx[i], normals.ny[i], normals.nz[i]
            ));
        }

        if let Some(ref colors) = cloud.colors {
            out.push_str(&format!(" {} {} {}", colors.r[i], colors.g[i], colors.b[i]));
        }

        out.push('\n');
    }

    fs::write(path, out)
}

/// Write a PLY file in binary_little_endian format.
///
/// Binary PLY is ~3-4x smaller and faster to read/write than ASCII PLY.
pub fn write_ply_binary(path: impl AsRef<Path>, cloud: &PointCloud) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);

    // Write ASCII header
    w.write_all(b"ply\n")?;
    w.write_all(b"format binary_little_endian 1.0\n")?;
    writeln!(w, "element vertex {}", cloud.len())?;
    w.write_all(b"property float x\n")?;
    w.write_all(b"property float y\n")?;
    w.write_all(b"property float z\n")?;

    if cloud.normals.is_some() {
        w.write_all(b"property float nx\n")?;
        w.write_all(b"property float ny\n")?;
        w.write_all(b"property float nz\n")?;
    }

    if cloud.colors.is_some() {
        w.write_all(b"property uchar red\n")?;
        w.write_all(b"property uchar green\n")?;
        w.write_all(b"property uchar blue\n")?;
    }

    w.write_all(b"end_header\n")?;

    // Write binary body
    for i in 0..cloud.len() {
        w.write_all(&cloud.x[i].to_le_bytes())?;
        w.write_all(&cloud.y[i].to_le_bytes())?;
        w.write_all(&cloud.z[i].to_le_bytes())?;

        if let Some(ref normals) = cloud.normals {
            w.write_all(&normals.nx[i].to_le_bytes())?;
            w.write_all(&normals.ny[i].to_le_bytes())?;
            w.write_all(&normals.nz[i].to_le_bytes())?;
        }

        if let Some(ref colors) = cloud.colors {
            w.write_all(&[colors.r[i]])?;
            w.write_all(&[colors.g[i]])?;
            w.write_all(&[colors.b[i]])?;
        }
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use tempfile::NamedTempFile;

    #[test]
    fn ply_roundtrip() {
        let cloud = PointCloud::from_xyz(
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        );
        let tmp = NamedTempFile::new().unwrap();
        write_ply(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.x, cloud.x);
        assert_eq!(loaded.y, cloud.y);
        assert_eq!(loaded.z, cloud.z);
        assert!(loaded.normals.is_none());
        assert!(loaded.colors.is_none());
    }

    #[test]
    fn ply_empty_cloud() {
        let cloud = PointCloud::new();
        let tmp = NamedTempFile::new().unwrap();
        write_ply(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn ply_roundtrip_with_normals() {
        let mut cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        cloud.normals = Some(Normals {
            nx: vec![0.0, 1.0],
            ny: vec![1.0, 0.0],
            nz: vec![0.0, 0.0],
        });
        let tmp = NamedTempFile::new().unwrap();
        write_ply(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let normals = loaded.normals.as_ref().unwrap();
        assert_eq!(normals.nx, vec![0.0, 1.0]);
        assert_eq!(normals.ny, vec![1.0, 0.0]);
        assert_eq!(normals.nz, vec![0.0, 0.0]);
    }

    #[test]
    fn ply_roundtrip_with_colors() {
        let mut cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        cloud.colors = Some(Colors {
            r: vec![255, 0],
            g: vec![0, 255],
            b: vec![128, 64],
        });
        let tmp = NamedTempFile::new().unwrap();
        write_ply(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let colors = loaded.colors.as_ref().unwrap();
        assert_eq!(colors.r, vec![255, 0]);
        assert_eq!(colors.g, vec![0, 255]);
        assert_eq!(colors.b, vec![128, 64]);
    }

    #[test]
    fn ply_binary_roundtrip() {
        let cloud = PointCloud::from_xyz(
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        );
        let tmp = NamedTempFile::new().unwrap();
        write_ply_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.x, cloud.x);
        assert_eq!(loaded.y, cloud.y);
        assert_eq!(loaded.z, cloud.z);
    }

    #[test]
    fn ply_binary_empty() {
        let cloud = PointCloud::new();
        let tmp = NamedTempFile::new().unwrap();
        write_ply_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn ply_binary_with_normals() {
        let mut cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        cloud.normals = Some(Normals {
            nx: vec![0.0, 1.0],
            ny: vec![1.0, 0.0],
            nz: vec![0.0, 0.0],
        });
        let tmp = NamedTempFile::new().unwrap();
        write_ply_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let normals = loaded.normals.as_ref().unwrap();
        assert_eq!(normals.nx, vec![0.0, 1.0]);
        assert_eq!(normals.ny, vec![1.0, 0.0]);
        assert_eq!(normals.nz, vec![0.0, 0.0]);
    }

    #[test]
    fn ply_binary_with_colors() {
        let mut cloud = PointCloud::from_xyz(vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]);
        cloud.colors = Some(Colors {
            r: vec![255, 0],
            g: vec![0, 255],
            b: vec![128, 64],
        });
        let tmp = NamedTempFile::new().unwrap();
        write_ply_binary(tmp.path(), &cloud).unwrap();
        let loaded = read_ply(tmp.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        let colors = loaded.colors.as_ref().unwrap();
        assert_eq!(colors.r, vec![255, 0]);
        assert_eq!(colors.g, vec![0, 255]);
        assert_eq!(colors.b, vec![128, 64]);
    }

    proptest! {
        #[test]
        fn ply_roundtrip_preserves_points(
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
            write_ply(tmp.path(), &cloud).unwrap();
            let loaded = read_ply(tmp.path()).unwrap();

            prop_assert_eq!(loaded.len(), cloud.len());
            for i in 0..cloud.len() {
                prop_assert_eq!(loaded.x[i], cloud.x[i]);
                prop_assert_eq!(loaded.y[i], cloud.y[i]);
                prop_assert_eq!(loaded.z[i], cloud.z[i]);
            }
        }

        #[test]
        fn ply_binary_roundtrip_preserves_data(
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
            write_ply_binary(tmp.path(), &cloud).unwrap();
            let loaded = read_ply(tmp.path()).unwrap();

            prop_assert_eq!(loaded.len(), cloud.len());
            for i in 0..cloud.len() {
                // Binary roundtrip should be bit-exact (no float parsing)
                prop_assert_eq!(loaded.x[i].to_bits(), cloud.x[i].to_bits());
                prop_assert_eq!(loaded.y[i].to_bits(), cloud.y[i].to_bits());
                prop_assert_eq!(loaded.z[i].to_bits(), cloud.z[i].to_bits());
            }
        }
    }
}
