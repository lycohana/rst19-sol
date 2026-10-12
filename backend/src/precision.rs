//! Statically dispatch the same detection algorithm over actual f32/f64 image storage.
use crate::{filtering::ProposalOptions, models::NumericPlane, Result};
pub trait Calculation:
    Copy + PartialEq + PartialOrd + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self>
{
    const FLOAT32: bool;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn finite(self) -> bool;
    fn plane(values: &[Self]) -> NumericPlane<'_>;
    fn stats(
        values: &[Self],
        mask: Option<&[bool]>,
        sigma: f64,
        iterations: usize,
        limit: usize,
    ) -> Result<(f64, f64)>;
    fn background(
        values: &[Self],
        mask: &[bool],
        h: usize,
        w: usize,
        options: crate::statistics::BackgroundOptions,
    ) -> Result<(Vec<Self>, Vec<Self>)>;
    fn gaussian(
        values: &[Self],
        valid: &[bool],
        h: usize,
        w: usize,
        sigma: f64,
        fast: bool,
    ) -> Result<(Vec<Self>, &'static str)>;
    fn starlet(
        values: &[Self],
        valid: &[bool],
        h: usize,
        w: usize,
        step: usize,
        fast: bool,
    ) -> Result<(Vec<Self>, &'static str)>;
    fn standardize(
        values: &[Self],
        mask: &[bool],
        noise: &[Self],
        valid: &[bool],
        limit: usize,
    ) -> Result<(Vec<Self>, f64, f64)>;
    fn maximum(values: &[Self], h: usize, w: usize, radius: usize) -> Result<Vec<Self>>;
    fn proposals(
        values: &[Self],
        valid: &[bool],
        h: usize,
        w: usize,
        options: &ProposalOptions,
        gate: Option<&[bool]>,
    ) -> Result<Vec<crate::filtering::Proposal>>;
    fn lines(
        values: &[Self],
        noise: &[Self],
        valid: &[bool],
        h: usize,
        w: usize,
        threshold: f64,
        fwhm: f64,
    ) -> Result<Vec<bool>>;
}
macro_rules! calculation {
    ($scalar:ty,$flag:expr,$stats:path,$background:path,$gaussian:path,$starlet:path,$standardize:path,$maximum:path,$proposals:path,$lines:path) => {
        impl Calculation for $scalar {
            const FLOAT32: bool = $flag;
            fn from_f64(v: f64) -> Self {
                v as Self
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn finite(self) -> bool {
                self.is_finite()
            }
            fn plane(v: &[Self]) -> NumericPlane<'_> {
                v.into()
            }
            fn stats(
                v: &[Self],
                m: Option<&[bool]>,
                s: f64,
                i: usize,
                l: usize,
            ) -> Result<(f64, f64)> {
                $stats(v, m, s, i, l)
            }
            fn background(
                v: &[Self],
                m: &[bool],
                h: usize,
                w: usize,
                o: crate::statistics::BackgroundOptions,
            ) -> Result<(Vec<Self>, Vec<Self>)> {
                $background(v, m, h, w, o)
            }
            fn gaussian(
                v: &[Self],
                m: &[bool],
                h: usize,
                w: usize,
                s: f64,
                f: bool,
            ) -> Result<(Vec<Self>, &'static str)> {
                $gaussian(v, m, h, w, s, f)
            }
            fn starlet(
                v: &[Self],
                m: &[bool],
                h: usize,
                w: usize,
                s: usize,
                f: bool,
            ) -> Result<(Vec<Self>, &'static str)> {
                $starlet(v, m, h, w, s, f)
            }
            fn standardize(
                v: &[Self],
                m: &[bool],
                n: &[Self],
                ok: &[bool],
                l: usize,
            ) -> Result<(Vec<Self>, f64, f64)> {
                $standardize(v, m, n, ok, l)
            }
            fn maximum(v: &[Self], h: usize, w: usize, r: usize) -> Result<Vec<Self>> {
                $maximum(v, h, w, r)
            }
            fn proposals(
                v: &[Self],
                m: &[bool],
                h: usize,
                w: usize,
                o: &ProposalOptions,
                g: Option<&[bool]>,
            ) -> Result<Vec<crate::filtering::Proposal>> {
                $proposals(v, m, h, w, o, g)
            }
            fn lines(
                v: &[Self],
                n: &[Self],
                m: &[bool],
                h: usize,
                w: usize,
                t: f64,
                f: f64,
            ) -> Result<Vec<bool>> {
                $lines(v, n, m, h, w, t, f)
            }
        }
    };
}
calculation!(
    f64,
    false,
    crate::statistics::sigma_clipped_stats,
    crate::statistics::local_background_rms,
    crate::filtering::masked_gaussian_response,
    crate::filtering::masked_starlet_smooth,
    crate::filtering::standardize_response,
    crate::filtering::maximum_filter,
    crate::filtering::response_proposals,
    crate::filtering::linear_artifact_mask
);
calculation!(
    f32,
    true,
    crate::statistics32::sigma_clipped_stats,
    crate::statistics32::local_background_rms,
    crate::filtering32::masked_gaussian_response,
    crate::filtering32::masked_starlet_smooth,
    crate::filtering32::standardize_response,
    crate::filtering32::maximum_filter,
    crate::filtering32::response_proposals,
    crate::filtering32::linear_artifact_mask
);
