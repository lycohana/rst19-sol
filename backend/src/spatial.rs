//! Exact 2-D spatial indexing for source association and proposal provenance.
use crate::Result;

#[derive(Debug)]
struct Node {
    point: [f64; 2],
    index: usize,
    axis: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}
#[derive(Debug)]
pub struct KdTree {
    root: Option<Box<Node>>,
}
impl KdTree {
    pub fn new(points: &[[f64; 2]]) -> Result<Self> {
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err("spatial coordinates must be finite".into());
        }
        fn build(rows: &mut [(usize, [f64; 2])], depth: usize) -> Option<Box<Node>> {
            if rows.is_empty() {
                return None;
            }
            let axis = depth % 2;
            let middle = rows.len() / 2;
            rows.select_nth_unstable_by(middle, |a, b| {
                a.1[axis].total_cmp(&b.1[axis]).then(a.0.cmp(&b.0))
            });
            let (index, point) = rows[middle];
            let (left, rest) = rows.split_at_mut(middle);
            Some(Box::new(Node {
                point,
                index,
                axis,
                left: build(left, depth + 1),
                right: build(&mut rest[1..], depth + 1),
            }))
        }
        let mut rows: Vec<_> = points.iter().copied().enumerate().collect();
        Ok(Self {
            root: build(&mut rows, 0),
        })
    }
    pub fn nearest(&self, point: [f64; 2]) -> Option<(usize, f64)> {
        self.nearest_ignoring(point, None)
    }

    /// Return the nearest point except for the supplied row index.
    pub fn nearest_excluding(&self, point: [f64; 2], excluded: usize) -> Option<(usize, f64)> {
        self.nearest_ignoring(point, Some(excluded))
    }

    fn nearest_ignoring(&self, point: [f64; 2], excluded: Option<usize>) -> Option<(usize, f64)> {
        if point.iter().any(|v| !v.is_finite()) {
            return None;
        }
        fn visit(node: &Node, p: [f64; 2], excluded: Option<usize>, best: &mut (usize, f64)) {
            let d = (p[0] - node.point[0]).powi(2) + (p[1] - node.point[1]).powi(2);
            if Some(node.index) != excluded && (d < best.1 || (d == best.1 && node.index < best.0))
            {
                *best = (node.index, d);
            }
            let delta = p[node.axis] - node.point[node.axis];
            let (near, far) = if delta < 0.0 {
                (&node.left, &node.right)
            } else {
                (&node.right, &node.left)
            };
            if let Some(near) = near {
                visit(near, p, excluded, best);
            }
            if delta * delta <= best.1 {
                if let Some(far) = far {
                    visit(far, p, excluded, best);
                }
            }
        }
        self.root
            .as_ref()
            .map(|root| {
                let mut best = (usize::MAX, f64::INFINITY);
                visit(root, point, excluded, &mut best);
                best
            })
            .and_then(|best| (best.0 != usize::MAX).then_some((best.0, best.1.sqrt())))
    }
    pub fn within_radius(&self, point: [f64; 2], radius: f64) -> Result<Vec<(usize, f64)>> {
        if point.iter().any(|v| !v.is_finite()) || !radius.is_finite() || radius < 0.0 {
            return Err("invalid spatial query".into());
        }
        fn visit(node: &Node, p: [f64; 2], sq: f64, out: &mut Vec<(usize, f64)>) {
            let d = (p[0] - node.point[0]).powi(2) + (p[1] - node.point[1]).powi(2);
            if d <= sq {
                out.push((node.index, d.sqrt()));
            }
            let delta = p[node.axis] - node.point[node.axis];
            if delta <= 0.0 || delta * delta <= sq {
                if let Some(left) = &node.left {
                    visit(left, p, sq, out);
                }
            }
            if delta >= 0.0 || delta * delta <= sq {
                if let Some(right) = &node.right {
                    visit(right, p, sq, out);
                }
            }
        }
        let mut out = Vec::new();
        if let Some(root) = &self.root {
            visit(root, point, radius * radius, &mut out);
        }
        out.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nearest_and_radius_queries_match_exhaustive_distances() {
        let points: Vec<_> = (0..1001)
            .map(|i| [((i * 37) % 4093) as f64, ((i * 13) % 521) as f64])
            .collect();
        let tree = KdTree::new(&points).unwrap();
        for p in [
            [0.0, 0.0],
            [1519.8, 292.8],
            [4095.0, 4095.0],
            [-100.0, 33.0],
        ] {
            let mut brute: Vec<_> = points
                .iter()
                .enumerate()
                .map(|(i, v)| (i, ((p[0] - v[0]).powi(2) + (p[1] - v[1]).powi(2)).sqrt()))
                .collect();
            brute.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(tree.nearest(p), Some(brute[0]));
            assert_eq!(
                tree.within_radius(p, 500.0).unwrap(),
                brute
                    .into_iter()
                    .filter(|(_, d)| *d <= 500.0)
                    .collect::<Vec<_>>()
            );
        }
        assert!(KdTree::new(&[[f64::NAN, 0.0]]).is_err());
        assert!(KdTree::new(&[]).unwrap().nearest([0.0, 0.0]).is_none());
    }

    #[test]
    fn nearest_excluding_skips_self_and_keeps_first_index_on_ties() {
        let points = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]];
        let tree = KdTree::new(&points).unwrap();
        assert_eq!(tree.nearest_excluding(points[0], 0), Some((3, 0.0)));
        assert_eq!(tree.nearest_excluding(points[1], 1), Some((0, 1.0)));
        assert_eq!(tree.nearest_excluding(points[2], 2), Some((0, 1.0)));
    }
}
