pub trait HasPosition {
    fn position(&self) -> [f32; 3];
}

pub trait HasColor {
    fn color(&self) -> [u8; 3];
}

pub trait HasNormal {
    fn normal(&self) -> [f32; 3];
}

pub trait HasIntensity {
    fn intensity(&self) -> f32;
}
