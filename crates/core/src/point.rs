#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointXYZRGB {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointXYZI {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointXYZNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}
