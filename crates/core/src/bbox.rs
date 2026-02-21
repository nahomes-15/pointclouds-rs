#[derive(Debug, Clone, PartialEq)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
    empty: bool,
}

impl Aabb {
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            empty: true,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.empty
    }

    pub fn expand_with_point(&mut self, point: [f32; 3]) {
        if !point.iter().all(|v| v.is_finite()) {
            return;
        }

        if self.empty {
            self.min = point;
            self.max = point;
            self.empty = false;
            return;
        }

        for (axis, &val) in point.iter().enumerate() {
            self.min[axis] = self.min[axis].min(val);
            self.max[axis] = self.max[axis].max(val);
        }
    }

    pub fn contains(&self, point: &[f32; 3]) -> bool {
        if self.empty || !point.iter().all(|v| v.is_finite()) {
            return false;
        }

        (0..3).all(|axis| point[axis] >= self.min[axis] && point[axis] <= self.max[axis])
    }

    pub fn from_xyz(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        let n = x.len().min(y.len()).min(z.len());
        let mut aabb = Self::empty();
        for i in 0..n {
            aabb.expand_with_point([x[i], y[i], z[i]]);
        }
        aabb
    }
}
