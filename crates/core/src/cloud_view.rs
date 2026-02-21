#[derive(Debug, Clone, Copy)]
pub struct CloudView<'a> {
    data: &'a [f32],
    num_points: usize,
}

impl<'a> CloudView<'a> {
    pub fn from_interleaved_xyz(data: &'a [f32], num_points: usize) -> Self {
        assert_eq!(
            data.len(),
            num_points * 3,
            "view source must have num_points * 3 floats"
        );
        Self { data, num_points }
    }

    pub fn len(&self) -> usize {
        self.num_points
    }

    pub fn is_empty(&self) -> bool {
        self.num_points == 0
    }

    pub fn point(&self, i: usize) -> [f32; 3] {
        assert!(i < self.num_points, "index out of bounds");
        let base = i * 3;
        [self.data[base], self.data[base + 1], self.data[base + 2]]
    }

    pub fn iter_points(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.data
            .chunks_exact(3)
            .take(self.num_points)
            .map(|c| [c[0], c[1], c[2]])
    }

    pub fn as_slice(&self) -> &'a [f32] {
        self.data
    }
}
