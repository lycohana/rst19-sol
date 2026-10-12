use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::path::PathBuf;

pub const AUXILIARY_FIELDS: [&str; 26] = [
    "p_az", "p_el", "ra", "dec", "j2000_x", "j2000_y", "j2000_z", "j2000_xv", "j2000_yv",
    "j2000_zv", "q1", "q2", "q3", "q4", "roll", "pitch", "yaw", "roll_v", "pitch_v", "yaw_v",
    "wgs84_x", "wgs84_y", "wgs84_z", "wgs84_xv", "wgs84_yv", "wgs84_zv",
];

#[derive(Debug, Clone, Copy)]
pub enum NumericPlane<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}
impl NumericPlane<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn value(self, index: usize) -> f64 {
        match self {
            Self::F64(v) => v[index],
            Self::F32(v) => v[index] as f64,
        }
    }
    pub fn is_float32(self) -> bool {
        matches!(self, Self::F32(_))
    }
}
impl<'a> From<&'a [f64]> for NumericPlane<'a> {
    fn from(value: &'a [f64]) -> Self {
        Self::F64(value)
    }
}
impl<'a> From<&'a [f32]> for NumericPlane<'a> {
    fn from(value: &'a [f32]) -> Self {
        Self::F32(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuxiliaryData {
    pub values: [f64; 26],
}
impl AuxiliaryData {
    pub fn as_dict(&self) -> Map<String, Value> {
        AUXILIARY_FIELDS
            .iter()
            .zip(self.values)
            .map(|(name, value)| ((*name).into(), Value::from(value)))
            .collect()
    }
    pub fn ra_deg(&self) -> f64 {
        self.values[2]
    }
    pub fn dec_deg(&self) -> f64 {
        self.values[3]
    }
}

/// Retain the exact FITS storage type. In particular i64 is not rounded via f64.
#[derive(Debug, Clone, Serialize)]
pub enum PixelData {
    U8(Vec<u8>),
    I16(Vec<i16>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}
impl PixelData {
    pub fn len(&self) -> usize {
        match self {
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::F64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn value(&self, index: usize) -> f64 {
        match self {
            Self::U8(v) => v[index] as f64,
            Self::I16(v) => v[index] as f64,
            Self::I32(v) => v[index] as f64,
            Self::I64(v) => v[index] as f64,
            Self::F32(v) => v[index] as f64,
            Self::F64(v) => v[index],
        }
    }
    pub fn exact_value(&self, index: usize) -> Value {
        match self {
            Self::U8(v) => Value::from(v[index]),
            Self::I16(v) => Value::from(v[index]),
            Self::I32(v) => Value::from(v[index]),
            Self::I64(v) => Value::from(v[index]),
            Self::F32(v) => Value::from(v[index] as f64),
            Self::F64(v) => Value::from(v[index]),
        }
    }
    pub fn dtype(&self) -> &'static str {
        match self {
            Self::U8(_) => "uint8",
            Self::I16(_) => ">i2",
            Self::I32(_) => ">i4",
            Self::I64(_) => ">i8",
            Self::F32(_) => ">f4",
            Self::F64(_) => ">f8",
        }
    }
    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.value(i)).collect()
    }
}

#[derive(Debug, Clone)]
pub struct FitsFrame {
    pub path: PathBuf,
    pub header: Map<String, Value>,
    pub data: PixelData,
    pub auxiliary: Option<AuxiliaryData>,
    pub data_offset: usize,
    pub width: usize,
    pub height: usize,
}
