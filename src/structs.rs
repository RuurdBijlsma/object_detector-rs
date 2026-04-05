use ndarray::Array2;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DetectorType {
    Promptable,
    PromptFree,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ModelScale {
    Nano,
    Small,
    Medium,
    Large,
    XLarge,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DetectedObject {
    pub bbox: ObjectBBox,
    pub score: f32,
    pub class_id: usize,
    pub tag: String,
    pub mask: Option<ObjectMask>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectBBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ObjectMask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl ObjectMask {
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> bool {
        let bit_idx = (y * self.width + x) as usize;
        self.data
            .get(bit_idx >> 3)
            .is_some_and(|&byte| (byte & (1 << (bit_idx & 7))) != 0)
    }

    #[must_use]
    pub fn to_array2(&self) -> Array2<bool> {
        Array2::from_shape_fn((self.height as usize, self.width as usize), |(y, x)| {
            self.get(x as u32, y as u32)
        })
    }
}
