//! cKDTree's compact, median-split, first-visited nearest-neighbor contract.
//! This separate index preserves legacy tie handling for scientific evidence selection.
use crate::Result;

enum Node {
    Leaf(Vec<usize>),
    Branch {
        axis: usize,
        split: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
}
pub struct CompatibleKdTree {
    points: Vec<[f64; 2]>,
    root: Option<Box<Node>>,
    mins: [f64; 2],
    maxes: [f64; 2],
}

fn compare(points: &[[f64; 2]], a: usize, b: usize, axis: usize) -> bool {
    points[a][axis] < points[b][axis]
}
fn move_median(
    rows: &mut [usize],
    points: &[[f64; 2]],
    result: usize,
    a: usize,
    b: usize,
    c: usize,
    axis: usize,
) {
    let cmp = |a, b| compare(points, rows[a], rows[b], axis);
    let chosen = if cmp(a, b) {
        if cmp(b, c) {
            b
        } else if cmp(a, c) {
            c
        } else {
            a
        }
    } else if cmp(a, c) {
        a
    } else if cmp(b, c) {
        c
    } else {
        b
    };
    rows.swap(result, chosen);
}
fn introselect(rows: &mut [usize], points: &[[f64; 2]], nth: usize, axis: usize) {
    let (mut first, mut last) = (0, rows.len());
    let mut depth = 2 * ((rows.len() as f64).log2().floor() as usize);
    while last - first > 3 {
        if depth == 0 {
            rows[first..last]
                .sort_unstable_by(|a, b| points[*a][axis].total_cmp(&points[*b][axis]));
            return;
        }
        depth -= 1;
        let middle = first + (last - first) / 2;
        move_median(rows, points, first, first + 1, middle, last - 1, axis);
        let pivot = rows[first];
        let (mut i, mut j) = (first + 1, last);
        loop {
            while compare(points, rows[i], pivot, axis) {
                i += 1;
            }
            j -= 1;
            while compare(points, pivot, rows[j], axis) {
                j -= 1;
            }
            if i >= j {
                break;
            }
            rows.swap(i, j);
            i += 1;
        }
        if i <= nth {
            first = i;
        } else {
            last = i;
        }
    }
    for i in first + 1..last {
        let value = rows[i];
        let mut j = i;
        while j > first && compare(points, value, rows[j - 1], axis) {
            rows[j] = rows[j - 1];
            j -= 1;
        }
        rows[j] = value;
    }
}
fn partition(rows: &mut [usize], points: &[[f64; 2]], axis: usize, pivot: f64) -> usize {
    let (mut first, mut last) = (0, rows.len());
    loop {
        while first < last && points[rows[first]][axis] < pivot {
            first += 1;
        }
        if first == last {
            return first;
        }
        loop {
            last -= 1;
            if first == last {
                return first;
            }
            if points[rows[last]][axis] < pivot {
                break;
            }
        }
        rows.swap(first, last);
        first += 1;
    }
}
fn bounds(rows: &[usize], points: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let mut mins = [f64::INFINITY; 2];
    let mut maxes = [f64::NEG_INFINITY; 2];
    for index in rows {
        for axis in 0..2 {
            mins[axis] = mins[axis].min(points[*index][axis]);
            maxes[axis] = maxes[axis].max(points[*index][axis]);
        }
    }
    (mins, maxes)
}
fn build(rows: &mut [usize], points: &[[f64; 2]]) -> Box<Node> {
    if rows.len() <= 16 {
        return Box::new(Node::Leaf(rows.to_vec()));
    }
    let (mins, maxes) = bounds(rows, points);
    let axis = if maxes[1] - mins[1] > maxes[0] - mins[0] {
        1
    } else {
        0
    };
    if maxes[axis] == mins[axis] {
        return Box::new(Node::Leaf(rows.to_vec()));
    }
    let middle = rows.len() / 2;
    introselect(rows, points, middle, axis);
    let mut split = points[rows[middle]][axis];
    let mut p = partition(&mut rows[..middle], points, axis, split);
    if p == 0 {
        split = mins[axis].next_up();
        p = partition(rows, points, axis, split);
    } else if p == rows.len() {
        split = maxes[axis];
        p = partition(rows, points, axis, split);
    }
    if p == 0 || p == rows.len() {
        return Box::new(Node::Leaf(rows.to_vec()));
    }
    let (left, right) = rows.split_at_mut(p);
    Box::new(Node::Branch {
        axis,
        split,
        left: build(left, points),
        right: build(right, points),
    })
}
struct Entry<'a> {
    node: &'a Node,
    min_distance: f64,
    sides: [f64; 2],
}
fn push<'a>(heap: &mut Vec<Entry<'a>>, entry: Entry<'a>) {
    heap.push(entry);
    let mut i = heap.len() - 1;
    while i > 0 && heap[i].min_distance < heap[(i - 1) / 2].min_distance {
        heap.swap(i, (i - 1) / 2);
        i = (i - 1) / 2;
    }
}
fn pop<'a>(heap: &mut Vec<Entry<'a>>) -> Option<Entry<'a>> {
    if heap.is_empty() {
        return None;
    }
    let last = heap.pop().unwrap();
    if heap.is_empty() {
        return Some(last);
    }
    let result = std::mem::replace(&mut heap[0], last);
    let mut i = 0;
    loop {
        let j = 2 * i + 1;
        let k = 2 * i + 2;
        if !(j < heap.len() && heap[i].min_distance > heap[j].min_distance
            || k < heap.len() && heap[i].min_distance > heap[k].min_distance)
        {
            break;
        }
        let child = if k < heap.len() && heap[j].min_distance > heap[k].min_distance {
            k
        } else {
            j
        };
        heap.swap(i, child);
        i = child;
    }
    Some(result)
}
impl CompatibleKdTree {
    pub fn new(points: &[[f64; 2]]) -> Result<Self> {
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err("spatial coordinates must be finite".into());
        }
        let mut rows: Vec<_> = (0..points.len()).collect();
        let (mins, maxes) = bounds(&rows, points);
        let root = if rows.is_empty() {
            None
        } else {
            Some(build(&mut rows, points))
        };
        Ok(Self {
            points: points.to_vec(),
            root,
            mins,
            maxes,
        })
    }
    pub fn nearest(&self, point: [f64; 2]) -> Option<(usize, f64)> {
        if point.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let root = self.root.as_ref()?;
        let sides = std::array::from_fn(|axis| {
            if point[axis] < self.mins[axis] {
                (self.mins[axis] - point[axis]).powi(2)
            } else if point[axis] > self.maxes[axis] {
                (point[axis] - self.maxes[axis]).powi(2)
            } else {
                0.0
            }
        });
        let mut current = Entry {
            node: root,
            min_distance: sides.iter().sum(),
            sides,
        };
        let mut heap = Vec::new();
        let mut best = None;
        let mut upper = f64::INFINITY;
        loop {
            match current.node {
                Node::Leaf(rows) => {
                    for index in rows {
                        let p = self.points[*index];
                        let d = (p[0] - point[0]).powi(2) + (p[1] - point[1]).powi(2);
                        if d < upper {
                            upper = d;
                            best = Some(*index);
                        }
                    }
                    let Some(next) = pop(&mut heap) else {
                        break;
                    };
                    current = next;
                }
                Node::Branch {
                    axis,
                    split,
                    left,
                    right,
                } => {
                    if current.min_distance > upper {
                        break;
                    }
                    let (near, far) = if point[*axis] < *split {
                        (left, right)
                    } else {
                        (right, left)
                    };
                    let side = (point[*axis] - split).powi(2);
                    let mut far_sides = current.sides;
                    let far_distance = current.min_distance + side - far_sides[*axis];
                    far_sides[*axis] = side;
                    if far_distance <= upper {
                        push(
                            &mut heap,
                            Entry {
                                node: far,
                                min_distance: far_distance,
                                sides: far_sides,
                            },
                        );
                    }
                    current.node = near;
                }
            }
        }
        best.map(|index| (index, upper.sqrt()))
    }
}
