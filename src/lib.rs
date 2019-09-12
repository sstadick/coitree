use std::cmp::max;
use std::cmp::Ordering::{self};
// small subtrees at the bottom of the tree are stored in sorted order. This gives the upper bound
// on hte size of such subtrees.
const SIMPLE_SUBTREE_CUTOFF: usize = 32;

#[derive(Clone, Debug)]
pub struct IntervalNode<T: Eq + Clone + std::fmt::Debug> {
    // subtree interval
    subtree_last: i32,
    // interval
    pub start: i32,
    pub stop: i32,
    // When this is the root of a simple subtree, left == right is the size of the subtree,
    // otherwise they are left, right child pointers
    left: u32,
    right: u32,

    pub val: T,
}
impl<T: Eq + Clone + std::fmt::Debug> IntervalNode<T> {
    /// Compute the intsect between two intervals
    #[inline]
    pub fn intersect(&self, other: &IntervalNode<T>) -> i32 {
        std::cmp::min(self.stop, other.stop)
            .checked_sub(std::cmp::max(self.start, other.start))
            .unwrap_or(0)
    }

    /// Check if two intervals overlap
    #[inline]
    pub fn overlap(&self, start: i32, stop: i32) -> bool {
        self.start < stop && self.stop > start
    }

    /// Check if a range overlaps the subtree of a node
    #[inline]
    pub fn overlap_subtree(&self, start: i32, stop: i32) -> bool {
        self.start < stop && self.subtree_last > start
    }

    pub fn new(start: i32, stop: i32, val: T) -> Self {
        IntervalNode {
            subtree_last: 0,
            start,
            stop,
            left: 0,
            right: 0,
            val,
        }
    }
}

impl<T: Eq + Clone + std::fmt::Debug> Ord for IntervalNode<T> {
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

impl<T: Eq + Clone + std::fmt::Debug> PartialOrd for IntervalNode<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<T: Eq + Clone + std::fmt::Debug> PartialEq for IntervalNode<T> {
    #[inline]
    fn eq(&self, other: &IntervalNode<T>) -> bool {
        self.start == other.start && self.stop == other.stop
    }
}

impl<T: Eq + Clone + std::fmt::Debug> Eq for IntervalNode<T> {}

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

#[derive(Debug)]
pub struct COITree<T: Eq + Clone + std::fmt::Debug> {
    nodes: Vec<IntervalNode<T>>,
}

// Find Iterator
#[derive(Debug)]
pub struct IterFind<'a, T>
where
    T: Eq + Clone + 'a + std::fmt::Debug,
{
    inner: &'a COITree<T>,
    results: Vec<&'a IntervalNode<T>>,
    off: usize,
    start: i32,
    stop: i32,
}

impl<'a, T: Eq + Clone + std::fmt::Debug> Iterator for IterFind<'a, T> {
    type Item = &'a IntervalNode<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.off < self.results.len() {
            self.off += 1;
            Some(self.results[self.off - 1])
        } else {
            None
        }
    }
}

// TODO: Move from max_value() to Option
// TODO: Move from clone to copy
impl<T: Eq + Clone + std::fmt::Debug> COITree<T> {
    pub fn new(nodes: Vec<IntervalNode<T>>) -> COITree<T> {
        COITree {
            nodes: Self::veb_order(nodes),
        }
    }

    pub fn find(&self, start: i32, stop: i32) -> IterFind<T> {
        IterFind {
            inner: self,
            results: self.query(start, stop),
            off: 0,
            start,
            stop,
        }
    }

    fn query_recursion<'a>(
        &'a self,
        root_idx: usize,
        start: i32,
        stop: i32,
        mut results: &mut Vec<&'a IntervalNode<T>>,
    ) {
        let node = &self.nodes[root_idx];
        if node.left == node.right {
            // simple subtree
            for k in root_idx..root_idx + node.right as usize {
                let node = &self.nodes[k];
                if node.overlap(start, stop) {
                    results.push(node);
                }
            }
        } else {
            if node.overlap(start, stop) {
                results.push(node);
            }
            let left = node.left as usize;
            if left < u32::max_value() as usize {
                if self.nodes[left].subtree_last >= start {
                    self.query_recursion(left, start, stop, &mut results);
                }
            }

            let right = node.right as usize;
            if right < u32::max_value() as usize {
                if node.overlap_subtree(start, stop) {
                    // check if it is down this subtree at all
                    self.query_recursion(right, start, stop, &mut results)
                }
            }
        }
    }

    fn query(&self, start: i32, stop: i32) -> Vec<&IntervalNode<T>> {
        let mut results: Vec<&IntervalNode<T>> = vec![];

        self.query_recursion(0, start, stop, &mut results);
        results
    }

    fn veb_order(mut nodes: Vec<IntervalNode<T>>) -> Vec<IntervalNode<T>> {
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

        let mut info = Self::traverse(&mut nodes);

        let max_depth = info.iter().fold(0, |md, info| max(info.depth, md));
        let idxs: &mut [usize] = &mut ((0..nodes.len()).collect::<Vec<usize>>());
        let tmp: &mut [usize] = &mut vec![0; nodes.len()];

        // put in dfs order
        for i in &*idxs {
            tmp[info[*i].dfs] = *i;
        }
        let (idxs, tmp) = (tmp, idxs);

        Self::veb_order_recursion(
            idxs,
            tmp,
            &mut info,
            0,
            nodes.len(),
            true,
            false,
            0,
            max_depth,
        );

        let revidx = tmp;
        for (i, j) in idxs.iter().enumerate() {
            revidx[*j] = i;
        }

        // put nodes in vEB order in a temp vector
        for i in 0..nodes.len() {
            veb_nodes[i] = nodes[idxs[i]].clone();

            if info[idxs[i]].simple {
                veb_nodes[i].left = info[idxs[i]].subtree_size as u32;
                veb_nodes[i].right = veb_nodes[i].left;
            } else {
                // update left and right pointers
                let left = info[idxs[i]].left as u32;
                veb_nodes[i].left = if left < u32::max_value() {
                    revidx[left as usize] as u32
                } else {
                    left
                };
                let right = info[idxs[i]].right as u32;
                veb_nodes[i].right = if right < u32::max_value() {
                    revidx[right as usize] as u32
                } else {
                    right
                };
            }
        }

        assert!(Self::compute_tree_size(&veb_nodes, 0) == veb_nodes.len());
        veb_nodes
    }

    // Recursively reorder indexes to put it in vEB order. Called by `veb_order`
    // idxs: current permutation
    // tmp: temporary space of equal length to idxs
    // nodes: the interval nodes (in sorted order)
    // start, end: slice within idxs to be reordered
    // childless: true if this slice is a proper subtree and has no children below it
    // parit: true if idxs and tmp are swapped and need to be copied back,
    // min_depth, max_depth: depth extreme of the start..end slice
    fn veb_order_recursion(
        idxs: &mut [usize],
        tmp: &mut [usize],
        info: &mut [TraversalInfo],
        start: usize,
        end: usize,
        childless: bool,
        parity: bool,
        min_depth: usize,
        max_depth: usize,
    ) {
        let n = (start..end).len();
        //let n = end - start; // TODO

        // small subtrees are just put in sorted order
        if childless && info[idxs[start]].subtree_size <= SIMPLE_SUBTREE_CUTOFF {
            assert!(n == info[idxs[start]].subtree_size);
            info[idxs[start]].simple = true;

            if parity {
                tmp[start..end].copy_from_slice(&idxs[start..end]);
            }
            return;
        }

        // very small trees are already in order
        if n <= 3 {
            if parity {
                tmp[start..end].copy_from_slice(&idxs[start..end]);
            }
            return;
        }

        let pivot_depth = min_depth + (max_depth - min_depth) / 2;
        let top_size = Self::stable_partition_by_depth(idxs, tmp, info, pivot_depth, start, end);

        // tmp is now partitioned by depth so swap pointers
        let (tmp, idxs) = (idxs, tmp);

        // recurse on top subtree
        Self::veb_order_recursion(
            idxs,
            tmp,
            info,
            start,
            start + top_size,
            false,
            !parity,
            min_depth,
            pivot_depth,
        );

        //find and recurse on bottom subtrees
        let bottom_subtree_depth = pivot_depth + 1;
        let mut i = start + top_size;
        while i < end {
            assert!(info[idxs[i]].depth == bottom_subtree_depth);
            let mut j = i + 1;
            let mut subtree_max_depth = info[idxs[i]].depth;
            while i < end && info[idxs[j]].depth != bottom_subtree_depth {
                assert!(info[idxs[j]].depth == bottom_subtree_depth);
                if info[idxs[j]].depth > subtree_max_depth {
                    subtree_max_depth = info[idxs[j]].depth;
                }
                j += 1;
            }
            Self::veb_order_recursion(
                idxs,
                tmp,
                info,
                i,
                j,
                childless,
                !parity,
                bottom_subtree_depth,
                subtree_max_depth,
            );
            i = j;
        }
    }

    // partition (in the quicksort sense) indexes according to the corresponding depths whil
    // retaining relative order.
    fn stable_partition_by_depth(
        input: &[usize],
        output: &mut [usize],
        info: &[TraversalInfo],
        pivot: usize,
        start: usize,
        end: usize,
    ) -> usize {
        let mut l = start;
        for i in start..end {
            if info[input[i]].depth <= pivot {
                output[l] = input[i];
                l += 1;
            }
        }

        let mut r = l;
        for i in start..end {
            if info[input[i]].depth > pivot {
                output[r] = input[i];
                r += 1;
            }
        }
        l - start
    }

    // Traverse the tree and return the size, used for a sanity check
    // TODO: move from u32::max_size() to Option<u32>
    fn compute_tree_size(nodes: &[IntervalNode<T>], root_idx: usize) -> usize {
        let mut subtree_size = 1;

        if nodes[root_idx].left == nodes[root_idx].right {
            subtree_size = nodes[root_idx].right as usize;
        } else {
            let left = nodes[root_idx].left as usize;
            if left < u32::max_value() as usize {
                subtree_size += Self::compute_tree_size(nodes, left);
            }
            let right = nodes[root_idx].right as usize;
            if right < u32::max_value() as usize {
                subtree_size += Self::compute_tree_size(nodes, right);
            }
        }
        subtree_size
    }

    // Simple two pass radix sort of 32bit integers (16 bits at a time) to sort nodes on start
    // position. tmp is temporary space for the first pass of equal length to nodes.
    fn radix_sort_nodes(nodes: &mut [IntervalNode<T>], tmp: &mut [IntervalNode<T>]) {
        //let max_fist = nodes
        //.iter()
        //.fold(0, |max_first, node| max(max_first, node.start));
        let mut count = 0;
        let n = nodes.len();

        const R: usize = 16;
        const K: usize = 0xffff + 1; // number of possible numbers in 16 bits
        const MASK: i32 = 0xffff; // 16 bits set to 1

        let mut shift: usize = 0;
        let mut radix_counts: Vec<u32> = vec![0; K];
        //let mut radix_counts: [u32: K] = [0; K];
        let mut from = nodes;
        let mut to = tmp;

        while count < 32 / R {
            //let mut radix_counts: [u32; K] = [0; K];
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
                to[radix_counts[radix] as usize] = from[i].clone();
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
        let n = nodes.len();
        let mut info = vec![
            TraversalInfo {
                depth: 0,
                dfs: 0,
                subtree_size: 0,
                left: 0,
                right: 0,
                simple: false
            };
            n
        ];
        let mut dfs = 0;
        Self::traverse_recursion(nodes, &mut info, 0, n, 0, &mut dfs);
        info
    }

    // The recursive part of the `traverse` function
    fn traverse_recursion(
        nodes: &mut [IntervalNode<T>],
        info: &mut [TraversalInfo],
        start: usize,
        end: usize,
        depth: usize,
        dfs: &mut usize,
    ) -> (usize, usize) {
        if start >= end {
            return (usize::max_value(), 0);
        }

        let root_idx = start + (end - start) / 2;
        info[root_idx].depth = depth;
        info[root_idx].dfs = *dfs;
        nodes[root_idx].subtree_last = nodes[root_idx].stop;
        *dfs += 1;
        let mut left = usize::max_value();
        let mut right = usize::max_value();
        let mut subtree_size = 1;

        if root_idx > start {
            let (left_, left_subtree_size) =
                Self::traverse_recursion(nodes, info, start, root_idx, depth + 1, dfs);
            left = left_;
            subtree_size += left_subtree_size;
            nodes[root_idx].subtree_last =
                max(nodes[root_idx].subtree_last, nodes[left].subtree_last);
        }

        if root_idx + 1 < end {
            let (right_, right_subtree_size) =
                Self::traverse_recursion(nodes, info, root_idx + 1, end, depth + 1, dfs);
            right = right_;
            subtree_size += right_subtree_size;
            nodes[root_idx].subtree_last =
                max(nodes[root_idx].subtree_last, nodes[right].subtree_last);
        }

        info[root_idx].subtree_size = subtree_size;
        info[root_idx].left = left;
        info[root_idx].right = right;
        (root_idx, subtree_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type Iv = IntervalNode<u32>;

    fn check_results(found: &Vec<&Iv>, expected: &Vec<&Iv>) {
        assert_eq!(found.len(), expected.len());
        for i in found {
            assert_eq!(
                expected
                    .iter()
                    .any(|e| e.start == i.start && e.stop == i.stop),
                true
            );
        }
    }
    fn setup_nonoverlapping() -> COITree<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(20)
            .map(|x| Iv::new(x, x + 10, 0))
            .collect();
        let tree = COITree::new(data);
        tree
    }
    fn setup_overlapping() -> COITree<u32> {
        let data: Vec<Iv> = (0..100)
            .step_by(10)
            .map(|x| Iv::new(x, x + 15, 0))
            .collect();
        let tree = COITree::new(data);
        tree
    }
    fn setup_badlapper() -> COITree<u32> {
        let data: Vec<Iv> = vec![
            Iv::new(70, 120, 0), // max_len = 50
            Iv::new(10, 15, 0),
            Iv::new(10, 15, 0), // exact overlap
            Iv::new(12, 15, 0), // inner overlap
            Iv::new(14, 16, 0), // overlap end
            Iv::new(40, 45, 0),
            Iv::new(50, 55, 0),
            Iv::new(60, 65, 0),
            Iv::new(68, 71, 0), // overlap start
            Iv::new(70, 75, 0),
        ];
        let tree = COITree::new(data);
        tree
    }
    fn setup_single() -> COITree<u32> {
        let data: Vec<Iv> = vec![Iv::new(10, 35, 0)];
        let tree = COITree::new(data);
        tree
    }

    // Test that a query stop that hits an interval start returns no interval
    #[test]
    fn test_query_stop_interval_start() {
        println!("query_stop_interval_start");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        assert_eq!(None, tree.find(15, 20).next());
        //assert_eq!(None, tree.seek(15, 20, &mut cursor).next())
    }

    // Test that a query start that hits an interval end returns no interval
    #[test]
    fn test_query_start_interval_stop() {
        println!("query_start_interval_stop");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        assert_eq!(None, tree.find(30, 35).next());
        //assert_eq!(None, tree.seek(30, 35, &mut cursor).next())
    }

    // Test that a query that overlaps the start of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_start() {
        println!("query_overlaps_interval_start");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv::new(20, 30, 0);
        assert_eq!(Some(&expected), tree.find(15, 25).next());
        //assert_eq!(Some(&expected), tree.seek(15, 25, &mut cursor).next())
    }
    // Test that a query that overlaps the stop of an interval returns that interval
    #[test]
    fn test_query_overlaps_interval_stop() {
        println!("query_overlaps_interval_stop");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv::new(20, 30, 0);
        assert_eq!(Some(&expected), tree.find(25, 35).next());
        //assert_eq!(Some(&expected), tree.seek(25, 35, &mut cursor).next())
    }

    // Test that a query that is enveloped by interval returns interval
    #[test]
    fn test_interval_envelops_query() {
        println!("interval_envelops_query");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv::new(20, 30, 0);
        assert_eq!(Some(&expected), tree.find(22, 27).next());
        //assert_eq!(Some(&expected), tree.seek(22, 27, &mut cursor).next())
    }

    // Test that a query that envolops an interval returns that interval
    #[test]
    fn test_query_envolops_interval() {
        println!("query_envolops_interval");
        let tree = setup_nonoverlapping();
        let mut cursor = 0;
        let expected = Iv::new(20, 30, 0);
        assert_eq!(Some(&expected), tree.find(15, 35).next());
        //assert_eq!(Some(&expected), tree.seek(15, 35, &mut cursor).next())
    }

    #[test]
    fn test_overlapping_intervals() {
        println!("overlapping_intervals");
        let tree = setup_overlapping();
        println!("Tree setup for overlapping intervals");
        let mut cursor = 0;
        let e1 = Iv::new(0, 15, 0);
        let e2 = Iv::new(10, 25, 0);
        println!("Checking for overlaps in overlapping intervals");
        let found = tree.find(8, 20).collect::<Vec<&Iv>>();
        let expected = vec![&e1, &e2];
        check_results(&found, &expected);
    }

    //#[test]
    //fn test_merge_overlaps() {
    //let mut tree = setup_badlapper();
    //let expected: Vec<&Iv> = vec![
    //&Iv( 10,  16,  0),
    //&Iv( 40,  45,  0),
    //&Iv( 50,  55,  0),
    //&Iv( 60,  65,  0),
    //&Iv( 68,  120,  0), // max_len = 50
    //];
    //tree.merge_overlaps();
    //assert_eq!(expected, tree.iter().collect::<Vec<&Iv>>())

    //}

    //#[test]
    //fn test_tree_cov() {
    //let mut tree = setup_badlapper();
    //let before = tree.cov();
    //tree.merge_overlaps();
    //let after = tree.cov();
    //assert_eq!(before, after);

    //let mut tree = setup_nonoverlapping();
    //tree.set_cov();
    //assert_eq!(tree.cov(), 50);
    //}

    //#[test]
    //fn test_interval_intersects() {
    //let i1 = Iv( 70,  120,  0); // max_len = 50
    //let i2 = Iv( 10,  15,  0);
    //let i3 = Iv( 10,  15,  0); // exact overlap
    //let i4 = Iv( 12,  15,  0); // inner overlap
    //let i5 = Iv( 14,  16,  0); // overlap end
    //let i6 = Iv( 40,  50,  0);
    //let i7 = Iv( 50,  55,  0);
    //let i_8 = Iv( 60,  65,  0);
    //let i9 = Iv( 68,  71,  0); // overlap start
    //let i10 = Iv( 70,  75,  0);

    //assert_eq!(i2.intersect(&i3), 5); // exact match
    //assert_eq!(i2.intersect(&i4), 3); // inner intersect
    //assert_eq!(i2.intersect(&i5), 1); // end intersect
    //assert_eq!(i9.intersect(&i10), 1); // start intersect
    //assert_eq!(i7.intersect(&i_8), 0); // no intersect
    //assert_eq!(i6.intersect(&i7), 0); // no intersect stop = start
    //assert_eq!(i1.intersect(&i10), 5); // inner intersect at start
    //}

    //#[test]
    //fn test_union_and_intersect() {
    //let data1: Vec<Iv> = vec![
    //Iv( 70,  120,  0), // max_len = 50
    //Iv( 10,  15,  0), // exact overlap
    //Iv( 12,  15,  0), // inner overlap
    //Iv( 14,  16,  0), // overlap end
    //Iv( 68,  71,  0), // overlap start
    //];
    //let data2: Vec<Iv> = vec![

    //Iv( 10,  15,  0),
    //Iv( 40,  45,  0),
    //Iv( 50,  55,  0),
    //Iv( 60,  65,  0),
    //Iv( 70,  75,  0),
    //];

    //let (mut tree1, mut lapper2) = (Lapper::new(data1), Lapper::new(data2)) ;
    //// Should be the same either way it's calculated
    //let (union, intersect) = tree1.union_and_intersect(&lapper2);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //let (union, intersect) = tree2.union_and_intersect(&lapper1);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //tree1.merge_overlaps();
    //tree1.set_cov();
    //tree2.merge_overlaps();
    //tree2.set_cov();

    //// Should be the same either way it's calculated
    //let (union, intersect) = tree1.union_and_intersect(&lapper2);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //let (union, intersect) = tree2.union_and_intersect(&lapper1);
    //assert_eq!(intersect, 10);
    //assert_eq!(union, 73);
    //}

    #[test]
    fn test_find_overlaps_in_large_intervals() {
        println!("find_overlaps_in_large_intervals");
        let data1: Vec<Iv> = vec![
            Iv::new(0, 8, 0),
            Iv::new(1, 10, 0),
            Iv::new(2, 5, 0),
            Iv::new(3, 8, 0),
            Iv::new(4, 7, 0),
            Iv::new(5, 8, 0),
            Iv::new(8, 8, 0),
            Iv::new(9, 11, 0),
            Iv::new(10, 13, 0),
            Iv::new(100, 200, 0),
            Iv::new(110, 120, 0),
            Iv::new(110, 124, 0),
            Iv::new(111, 160, 0),
            Iv::new(150, 200, 0),
        ];
        let tree = COITree::new(data1);
        let found = tree.find(8, 11).collect::<Vec<&Iv>>();
        check_results(
            &found,
            &vec![&Iv::new(1, 10, 0), &Iv::new(9, 11, 0), &Iv::new(10, 13, 0)],
        );
        let found = tree.find(145, 151).collect::<Vec<&Iv>>();
        check_results(
            &found,
            &vec![
                &Iv::new(100, 200, 0),
                &Iv::new(111, 160, 0),
                &Iv::new(150, 200, 0),
            ],
        );
    }

    //#[test]
    //fn test_depth_sanity() {
    //let data1: Vec<Iv> = vec![
    //Iv( 0,  10,  0),
    //Iv( 5,  10,  0}
    //];
    //let tree = Lapper::new(data1);
    //let found = tree.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{ 0,  5,  1),
    //Interval{ 5,  10,  2}
    //]);
    //}

    //#[test]
    //fn test_depth_hard() {
    //let data1: Vec<Iv> = vec![
    //Iv( 1,  10,  0),
    //Iv( 2,  5,  0),
    //Iv( 3,  8,  0),
    //Iv( 3,  8,  0),
    //Iv( 3,  8,  0),
    //Iv( 5,  8,  0),
    //Iv( 9,  11,  0),
    //];
    //let tree = Lapper::new(data1);
    //let found = tree.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{ 1,  2,  1),
    //Interval{ 2,  3,  2),
    //Interval{ 3,  8,  5),
    //Interval{ 8,  9,  1),
    //Interval{ 9,  10,  2),
    //Interval{ 10,  11,  1),
    //]);
    //}
    //#[test]
    //fn test_depth_harder() {
    //let data1: Vec<Iv> = vec![
    //Iv( 1,  10,  0),
    //Iv( 2,  5,  0),
    //Iv( 3,  8,  0),
    //Iv( 3,  8,  0),
    //Iv( 3,  8,  0),
    //Iv( 5,  8,  0),
    //Iv( 9,  11,  0),
    //Iv( 15,  20,  0),
    //];
    //let tree = Lapper::new(data1);
    //let found = tree.depth().collect::<Vec<Interval<u32>>>();
    //assert_eq!(found, vec![
    //Interval{ 1,  2,  1),
    //Interval{ 2,  3,  2),
    //Interval{ 3,  8,  5),
    //Interval{ 8,  9,  1),
    //Interval{ 9,  10,  2),
    //Interval{ 10,  11,  1),
    //Interval{ 15,  20,  1),
    //]);
    //}
    // BUG TESTS - these are tests that came from real life

    // Test that it's not possible to induce index out of bounds by pushing the cursor past the end
    // of the tree.
    //#[test]
    //fn test_seek_over_len() {
    //let tree = setup_nonoverlapping();
    //let single = setup_single();
    //let mut cursor: usize = 0;

    //for interval in tree.iter() {
    //for o_interval in single.seek(interval.start, interval.stop, &mut cursor) {
    //println!("{:#?}", o_interval);
    //}
    //}
    //}

    // Test that if lower_bound puts us before the first match, we still return a match
    #[test]
    fn test_find_over_behind_first_match() {
        println!("find_over_behind_first_match");
        let tree = setup_badlapper();
        let e1 = Iv::new(50, 55, 0);
        let found = tree.find(50, 55).next();
        assert_eq!(found, Some(&e1));
    }

    // When there is a very long interval that spans many little intervals, test that the little
    // intevals still get returne properly
    #[test]
    fn test_bad_skips() {
        println!("bad_skips");
        let data = vec![
            Iv::new(25264912, 25264986, 0),
            Iv::new(27273024, 27273065, 0),
            Iv::new(27440273, 27440318, 0),
            Iv::new(27488033, 27488125, 0),
            Iv::new(27938410, 27938470, 0),
            Iv::new(27959118, 27959171, 0),
            Iv::new(28866309, 33141404, 0),
        ];
        let tree = COITree::new(data);

        let found = tree.find(28974798, 33141355).collect::<Vec<&Iv>>();
        check_results(&found, &vec![&Iv::new(28866309, 33141404, 0)]);
    }
}
