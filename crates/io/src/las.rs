use pointclouds_core::PointCloud;
use std::io;
use std::path::Path;

pub fn read_las(path: impl AsRef<Path>) -> io::Result<PointCloud> {
    let mut reader = las::Reader::from_path(path.as_ref())
        .map_err(|e| io::Error::other(format!("failed to open LAS file: {}", e)))?;

    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    let mut intensity = Vec::new();
    let mut has_nonzero_intensity = false;

    for point_result in reader.points() {
        let point = point_result.map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to read LAS point: {}", e),
            )
        })?;
        x.push(point.x as f32);
        y.push(point.y as f32);
        z.push(point.z as f32);
        let i = point.intensity as f32;
        if point.intensity != 0 {
            has_nonzero_intensity = true;
        }
        intensity.push(i);
    }

    let mut cloud = PointCloud::from_xyz(x, y, z);
    if has_nonzero_intensity {
        cloud.intensity = Some(intensity);
    }

    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn las_read_nonexistent() {
        let result = read_las("/tmp/nonexistent_file_that_does_not_exist_12345.las");
        assert!(result.is_err());
    }

    #[test]
    fn las_roundtrip_via_writer() {
        use las::Writer;
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path();

        // Write a minimal LAS file using the `las` crate directly
        let mut builder = las::Builder::from((1, 2));
        builder.point_format = las::point::Format::new(0).unwrap();
        let header = builder.into_header().unwrap();
        let mut writer = Writer::from_path(path, header).unwrap();

        let mut p1 = las::point::Point::default();
        p1.x = 1.0;
        p1.y = 2.0;
        p1.z = 3.0;
        p1.intensity = 100;
        writer.write(p1).unwrap();

        let mut p2 = las::point::Point::default();
        p2.x = 4.0;
        p2.y = 5.0;
        p2.z = 6.0;
        p2.intensity = 200;
        writer.write(p2).unwrap();

        drop(writer);

        // Read it back through our read_las
        let cloud = read_las(path).unwrap();
        assert_eq!(cloud.len(), 2);
        assert!((cloud.x[0] - 1.0).abs() < 0.01);
        assert!((cloud.y[0] - 2.0).abs() < 0.01);
        assert!((cloud.z[0] - 3.0).abs() < 0.01);
        assert!((cloud.x[1] - 4.0).abs() < 0.01);

        // Should have intensity
        assert!(cloud.intensity.is_some());
        let intensity = cloud.intensity.as_ref().unwrap();
        assert_eq!(intensity[0], 100.0);
        assert_eq!(intensity[1], 200.0);
    }
}
