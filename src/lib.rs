use std::cmp::Ordering::{self};
// small subtrees at the bottom of the tree are stored in sorted order. This gives the upper bound
// on hte size of such subtrees.
const SIMPLE_SUBTREE_CUTOFF: usize = 32;

#[derive(Clone, Debug)]
struct IntervalNode<T: Eq + Clone> {
    // subtree interval
    subtree_last: i32,
    // interval
    start: i32,
    stop: i32,
    // When this is the root of a simple subtree, left == right is the size of the subtree,
    // otherwise they are left, right child pointers
    left: u32,
    right: u32,

    metadata: T,
}
//impl<T: Eq + Clone> Interval<T> {
///// Compute the intsect between two intervals
//#[inline]
//pub fn intersect(&self, other: &Interval<T>) -> usize {
//std::cmp::min(self.stop, other.stop)
//.checked_sub(std::cmp::max(self.start, other.start))
//.unwrap_or(0)
//}

///// Check if two intervals overlap
//#[inline]
//pub fn overlap(&self, start: usize, stop: usize) -> bool {
//self.start < stop && self.stop > start
//}
//}

impl<T: Eq + Clone> Ord for IntervalNode<T> {
    #[inline]
    fn cmp(&self, other: &IntervalNode<T>) -> Ordering {
        if self.start < other.start {
            Ordering::Less
        } else if other.start < self.start {
            Ordering::Greater
        } else {
            self.stop.cmp(&other.stop)
        }
    }
}

impl<T: Eq + Clone> PartialOrd for IntervalNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<T: Eq + Clone> PartialEq for IntervalNode<T> {
    #[inline]
    fn eq(&self, other: &IntervalNode<T>) -> bool {
        self.start == other.start && self.stop == other.stop
    }
}

impl<T: Eq + Clone> Eq for IntervalNode<T> {}

// used by `traverse` to record tree metadata
#[derive(Clone, Debug)]
struct TraversalInfo {
    depth: usize,
    dfs: usize,
    subtree_size: usize,
    left: usize,
    right: usize,
    simple: bool, // set by veb_order_recursion
}

// dfs traversal of an implicit bst computing dfs number, node depth, subtree size, and left and
// right pointers.
fn traverse<T: Eq, Clone

struct COITree<T: Eq + Clone> {
    nodes: Vec<IntervalNode<T>>,
}

impl<T: Eq + Clone> COITree<T> {
    pub fn new(nodes: Vec<IntervalNode<T>>) -> COITree<T> {
        COITree {
            nodes: Self::veb_order(nodes),
        }
    }

    fn veb_order(nodes: Vec<IntervalNode<T>>) -> Vec<IntervalNode<T>> {
        let mut veb_nodes = nodes.clone();
        let mut nodes_presorted = true;
        for i in 1..nodes.len() {
            if nodes[i] < nodes[i - 1] {
                nodes_presorted = false;
                break;
            }
        }

        if !nodes_presorted {
            Self::radix_sort_nodes(&mut nodes, &mut veb_nodes);
        }

        let mut info = traverse(&mut nodes);

        let max_depth = info.iter().fold(0, |md, info| max(info.depth, md));
        let idx: &mut[usize] = &mut ((0..nodes.len()).collect::<Vec<usize>>());
        let tmp: &mut[usize] = &mut vec![0; nodes.len()];

        // put in dfs order
        for i in idxs {
            tmp[info[i].dfs] = i;
        }
        let (idxs, tmp) = (tmp, idxs);

        // line 352
    }

    // Simple two pass radix sort of 32bit integers (16 bits at a time) to sort nodes on start
    // position. tmp is temporary space for the first pass of equal length to nodes.
    fn radix_sort_nodes(nodes: &mut [IntervalNode<T>], tmp: &mut [IntervalNode<T>]) {
        let max_fist = nodes
            .iter()
            .fold(0, |max_first, node| max(max_first, node.start));
        let mut count = 0;
        let n = nodes.len();

        const R: usize = 16;
        const K: usize = 0xffff + 1;
        const MASK: i32 = 0xffff;

        let mut shift: usize = 0;
        let mut radix_counts: Vec<u32> = vec![0; K];

        let mut from = nodes;
        let mut to = tmp;

        while count < 32 / R {
            for i in 0..K {
                radix_counts[i] = 0;
            }

            for i in 0..n {
                radix_counts[((from[i].start >> shift) & MASK) as usize] += 1;
            }

            // make counts cumlative
            for i in 1..K {
                radix_counts[i] += radix_counts[i - 1];
            }

            // change counts to offsets
            for i in 0..K - 1 {
                radix_counts[K - 1 - i] = radix_counts[K - 2 - i];
            }
            radix_counts[0] = 0;

            for i in 0..n {
                let radix = ((from[i].start >> shift) & MASK) as usize;
                to[radix_counts[radix] as usize] = from[i];
                radix_counts[radix] += 1;
            }
            count += 1;
            shift += 1;

            let swap_tmp = from;
            from = to;
            to = swap_tmp;
        }
    }
    // dfs traversal of an implicit bst computing dfs number, node depth, subtree size, and left and
    // right pointers.
    fn traverse(nodes: &mut [IntervalNode<T>]) -> Vec<TraversalInfo> {
        let n = nodes.len()
            let mut info = vec![TraversalInfo{
                depth: 0, dfs: 0, subtree_size: 0, left: 0, right: 0, simple: false}; n];
            let mut dfs = 0;
            traverse_recursion(nodes, &mut info, 0, n, 0, &mut dfs);
            info
    }

    // The recursive part of the `traverse` function 
    fn traverse_recursion(nodes: &mut [IntervalNode<T>], info: &mut [TraversalInfo], start: usize, end: usize, depth: usize, dfs: &mut usize) -> (usize, usize) {
        if start >= end {
            return (usize::max_value(), 0);
        }

        let root_idx = start + (end - start) / 2;
        info[root_idx].depth = depth;
        info[root_idx].dfs = *dfs;
        nodes[root_idx].subtree_last = nodes[root_idx].last;
        *dfs += 1;
        let mut left = usize::max_value();
        let mut right = usize::max_value();
        let mut subtree_size = 1;

        if root_idx > start {
            let (left_, left_subtree_size) = traverse_recursion(nodes, info, start, root_idx, depth+1, dfs);
            left = left_;
            subtree_size += left_subtree_size;
            nodes[root_idx].subtree_last = max(nodes[root_idx].subtree_last, nodes[left].subtree_last);
        }

        if root_idx + 1 < end {
            let (right_, right_subtree_size) = traverse_recursion(nodes, info, root_idx + 1, end, depth + 1, dfs);
            right = right_;
            subtree_size += right_subtree_size;
            nodes[root_idx].subtree_last = max(nodes[root_idx].subtree_last, nodes[right].subtree_last);
        }

        info[root_idx].subtree_size = subtree_size;
        info[root_idx].left = left;
        inof[root_idx].right = right;
        (root_idx, subtree_size)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
