use std::fmt::Debug;
use std::marker::PhantomData;

use crate::util_types::simple_hasher::{Hasher, ToDigest};

use super::other::log_2_floor;

#[inline]
fn left_child(node_index: usize, height: usize) -> usize {
    node_index - (1 << height)
}

#[inline]
fn right_child(node_index: usize) -> usize {
    node_index - 1
}

/// Get (index, height) of leftmost ancestor
// This ancestor does *not* have to be in the MMR
fn leftmost_ancestor(node_index: usize) -> (usize, usize) {
    let mut h = 0;
    let mut ret = 1;
    while ret < node_index {
        h += 1;
        ret = (1 << (h + 1)) - 1;
    }

    (ret, h)
}

/// Return the tuple: (is_right_child, height)
fn right_child_and_height(node_index: usize) -> (bool, usize) {
    // 1. Find leftmost_ancestor(n), if leftmost_ancestor(n) == n => left_child (false)
    // 2. Let node = leftmost_ancestor(n)
    // 3. while(true):
    //    if n == left_child(node):
    //        return false
    //    if n < left_child(node):
    //        node = left_child(node)
    //    if n == right_child(node):
    //        return true
    //    else:
    //        node = right_child(node);

    // 1.
    let (leftmost_ancestor, ancestor_height) = leftmost_ancestor(node_index);
    if leftmost_ancestor == node_index {
        return (false, ancestor_height);
    }

    let mut node = leftmost_ancestor;
    let mut height = ancestor_height;
    loop {
        let left_child = left_child(node, height);
        height -= 1;
        if node_index == left_child {
            return (false, height);
        }
        if node_index < left_child {
            node = left_child;
        } else {
            let right_child = right_child(node);
            if node_index == right_child {
                return (true, height);
            }
            node = right_child;
        }
    }
}

/// Get the node_index of the parent
fn parent(node_index: usize) -> usize {
    let (right, height) = right_child_and_height(node_index);

    if right {
        node_index + 1
    } else {
        node_index + (1 << (height + 1))
    }
}

#[inline]
fn left_sibling(node_index: usize, height: usize) -> usize {
    node_index - (1 << (height + 1)) + 1
}

#[inline]
fn right_sibling(node_index: usize, height: usize) -> usize {
    node_index + (1 << (height + 1)) - 1
}

fn get_height_from_data_index(data_index: usize) -> usize {
    log_2_floor(data_index as u64 + 1) as usize
}

/// Count the number of non-leaf nodes that were inserted *prior* to
/// the insertion of this leaf.
fn non_leaf_nodes_left(data_index: usize) -> usize {
    if data_index == 0 {
        return 0;
    }

    let mut acc = 0;
    let mut data_index_acc = data_index;
    while data_index_acc > 0 {
        // Accumulate how many nodes in the tree of the nearest left neighbor that are not leafs.
        // We count this number for the nearest left neighbor since only the non-leafs in that
        // tree were inserted prior to the leaf this function is called for.
        // For a tree of height 2, there are 2^2 - 1 non-leaf nodes, note that height starts at
        // 0.
        // Since more than one subtree left of the requested index can contain non-leafs, we have
        // to run this accumulater untill data_index_acc is zero.
        let left_data_height = get_height_from_data_index(data_index_acc - 1);
        acc += (1 << left_data_height) - 1;
        data_index_acc -= 1 << left_data_height;
    }

    acc
}

pub fn data_index_to_node_index(data_index: usize) -> usize {
    let diff = non_leaf_nodes_left(data_index);

    data_index + diff + 1
}

/// Convert from node index to data index in log(size) time
pub fn node_index_to_data_index(node_index: usize) -> Option<usize> {
    let (_right, height) = right_child_and_height(node_index);
    if height != 0 {
        return None;
    }

    let (mut node, mut height) = leftmost_ancestor(node_index);
    let mut data_index = 0;
    while height > 0 {
        let left_child = left_child(node, height);
        if node_index <= left_child {
            node = left_child;
            height -= 1;
        } else {
            node = right_child(node);
            height -= 1;
            data_index += 1 << height;
        }
    }

    Some(data_index)
}

#[derive(Debug, Clone)]
pub struct LightMmr<HashDigest, H> {
    leaf_count: u128,
    peaks: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

impl<HashDigest, H> LightMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest>,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
{
    /// Initialize a shallow MMR (only storing peaks) from a list of hash digests
    pub fn from_leafs(hashes: Vec<HashDigest>) -> Self {
        // If all the hash digests already exist in memory, we might as well
        // build the shallow MMR from an archival MMR, since it doesn't give
        // asymptotically higher RAM consumption.
        let archival = ArchivalMmr::init(hashes, zero);
    }

    pub fn prove_append() {
        todo!()
    }

    pub fn verify_append() {
        todo!()
    }

    pub fn prove_modify() {
        todo!()
    }

    pub fn verify_modify() {
        todo!()
    }

    pub fn prove_membership() {
        todo!()
    }

    pub fn verify_membership() {
        todo!()
    }
}

/// A Merkle Mountain Range is a datastructure for storing a list of hashes.
///
/// Merkle Mountain Ranges only know about hashes. When values are to be associated with
/// MMRs, these values must be stored by the caller, or in a wrapper to this data structure.
#[derive(Debug, Clone)]
pub struct ArchivalMmr<HashDigest, H> {
    digests: Vec<HashDigest>,
    _hasher: PhantomData<H>,
}

impl<HashDigest, H> ArchivalMmr<HashDigest, H>
where
    H: Hasher<Digest = HashDigest>,
    HashDigest: ToDigest<HashDigest> + PartialEq + Clone + Debug,
    u128: ToDigest<HashDigest>,
{
    pub fn init(hashes: Vec<HashDigest>, zero: HashDigest) -> Self {
        let mut new_mmr: Self = Self {
            digests: vec![zero.to_digest()],
            _hasher: PhantomData,
        };
        for hash in hashes {
            new_mmr.archive_append(hash);
        }

        new_mmr
    }

    pub fn verify_membership(
        root: &HashDigest,
        authentication_path: &[HashDigest],
        peaks: &[HashDigest],
        node_count: u128,
        value_hash: HashDigest,
        data_index: usize,
    ) -> bool {
        // Verify that peaks match root
        let matching_root = *root == Self::get_root_from_peaks(peaks, node_count);
        let node_index = data_index_to_node_index(data_index);

        let mut hasher = H::new();
        let mut acc_hash: HashDigest = value_hash;
        let mut acc_index: usize = node_index;
        for hash in authentication_path.iter() {
            let (acc_right, _acc_height) = right_child_and_height(acc_index);
            acc_hash = if acc_right {
                hasher.hash_two(hash, &acc_hash)
            } else {
                hasher.hash_two(&acc_hash, hash)
            };
            acc_index = parent(acc_index);
        }

        peaks.iter().any(|peak| *peak == acc_hash) && matching_root
    }

    /// Return (authentication_path, peaks)
    pub fn prove_membership(&self, data_index: usize) -> (Vec<HashDigest>, Vec<HashDigest>) {
        // A proof consists of an authentication path
        // and a list of peaks that must hash to the root

        // Find out how long the authentication path is
        let node_index = data_index_to_node_index(data_index);
        let mut top_height: i32 = -1;
        let mut parent_index = node_index;
        while parent_index < self.digests.len() {
            parent_index = parent(parent_index);
            top_height += 1;
        }

        // Build the authentication path
        let mut authentication_path: Vec<HashDigest> = vec![];
        let mut index = node_index;
        let (mut index_is_right_child, mut index_height) = right_child_and_height(index);
        while index_height < top_height as usize {
            if index_is_right_child {
                let left_sibling_index = left_sibling(index, index_height);
                authentication_path.push(self.digests[left_sibling_index].clone());
            } else {
                let right_sibling_index = right_sibling(index, index_height);
                authentication_path.push(self.digests[right_sibling_index].clone());
            }
            index = parent(index);
            let next_index_info = right_child_and_height(index);
            index_is_right_child = next_index_info.0;
            index_height = next_index_info.1;
        }

        let peaks: Vec<HashDigest> = self
            .get_peaks_with_heights()
            .iter()
            .map(|x| x.0.clone())
            .collect();

        (authentication_path, peaks)
    }

    /// Calculate root from a list of peaks and from the node count
    fn get_root_from_peaks(peaks: &[HashDigest], node_count: u128) -> HashDigest {
        // Follows the description for "bagging" on
        // https://github.com/mimblewimble/grin/blob/master/doc/mmr.md#hashing-and-bagging
        // Note that their "size" is the node count
        let peaks_count: usize = peaks.len();
        let mut hasher: H = H::new();

        let mut acc: HashDigest = hasher.hash_two(&node_count.to_digest(), &peaks[peaks_count - 1]);
        for i in 1..peaks_count {
            acc = hasher.hash_two(&peaks[peaks_count - 1 - i], &acc);
        }

        acc
    }

    /// Calculate the root for the entire MMR
    pub fn bag_peaks(&self) -> HashDigest {
        let peaks: Vec<HashDigest> = self
            .get_peaks_with_heights()
            .iter()
            .map(|x| x.0.clone())
            .collect();

        Self::get_root_from_peaks(&peaks, self.count_nodes() as u128)
    }

    /// Return a list of tuples (peaks, height)
    pub fn get_peaks_with_heights(&self) -> Vec<(HashDigest, usize)> {
        // 1. Find top peak
        // 2. Jump to right sibling (will not be included)
        // 3. Take left child of sibling, continue until a node in tree is found
        // 4. Once new node is found, jump to right sibling (will not be included)
        // 5. Take left child of sibling, continue until a node in tree is found
        let mut peaks_and_heights: Vec<(HashDigest, usize)> = vec![];
        let (mut top_peak, mut top_height) = leftmost_ancestor(self.digests.len() - 1);
        if top_peak > self.digests.len() - 1 {
            top_peak = left_child(top_peak, top_height);
            top_height -= 1;
        }
        peaks_and_heights.push((self.digests[top_peak].clone(), top_height)); // No clone needed bc array
        let mut height = top_height;
        let mut candidate = right_sibling(top_peak, height);
        'outer: while height > 0 {
            '_inner: while candidate > self.digests.len() && height > 0 {
                candidate = left_child(candidate, height);
                height -= 1;
                if candidate < self.digests.len() {
                    peaks_and_heights.push((self.digests[candidate].clone(), height));
                    candidate = right_sibling(candidate, height);
                    continue 'outer;
                }
            }
        }

        peaks_and_heights
    }

    pub fn count_nodes(&self) -> u128 {
        self.digests.len() as u128 - 1
    }

    /// Return the number of leaves in the tree
    pub fn count_leaves(&self) -> usize {
        let peaks_and_heights: Vec<(_, usize)> = self.get_peaks_with_heights();
        let mut acc = 0;
        for (_, height) in peaks_and_heights {
            acc += 1 << height
        }

        acc
    }

    fn archive_append(&mut self, hash: HashDigest) {
        let node_index = self.digests.len();
        self.digests.push(hash.clone());
        let (parent_needed, own_height) = right_child_and_height(node_index);
        if parent_needed {
            let left_sibling_hash = self.digests[left_sibling(node_index, own_height)].clone();
            let mut hasher = H::new();
            let parent_hash: HashDigest = hasher.hash_two(&left_sibling_hash, &hash);
            self.archive_append(parent_hash);
        }
    }

    /// With knowledge of old peaks, old size (leaf count), new leaf hash, and new peaks, verify that
    /// append is correct.
    pub fn verify_append(
        old_root: HashDigest,
        old_peaks: &[HashDigest],
        old_leaf_count: usize,
        new_root: HashDigest,
        new_leaf_hash: HashDigest,
        new_peaks: &[HashDigest],
    ) -> bool {
        let first_new_node_index = data_index_to_node_index(old_leaf_count);
        let (mut new_node_is_right_child, _height) = right_child_and_height(first_new_node_index);

        // If new node is not a right child, the new peak list is just the old one
        // with the new leaf hash appended
        let mut calculated_peaks: Vec<HashDigest> = old_peaks.to_vec();
        calculated_peaks.push(new_leaf_hash);
        let mut new_node_index = first_new_node_index;
        let mut hasher = H::new();
        while new_node_is_right_child {
            let new_hash = calculated_peaks.pop().unwrap();
            let previous_peak = calculated_peaks.pop().unwrap();
            calculated_peaks.push(hasher.hash_two(&previous_peak, &new_hash));
            new_node_index += 1;
            new_node_is_right_child = right_child_and_height(new_node_index).0;
        }

        let calculated_new_root =
            Self::get_root_from_peaks(&calculated_peaks, new_node_index as u128);
        let calculated_old_root =
            Self::get_root_from_peaks(old_peaks, first_new_node_index as u128 - 1);

        calculated_peaks == new_peaks
            && calculated_new_root == new_root
            && calculated_old_root == old_root
    }
}

#[cfg(test)]
mod mmr_test {
    use itertools::izip;
    use rand::RngCore;

    use super::*;
    use crate::{
        shared_math::{
            b_field_element::BFieldElement, rescue_prime::RescuePrime, rescue_prime_params,
        },
        util_types::simple_hasher::RescuePrimeProduction,
    };

    #[test]
    fn data_index_to_node_index_test() {
        assert_eq!(1, data_index_to_node_index(0));
        assert_eq!(2, data_index_to_node_index(1));
        assert_eq!(4, data_index_to_node_index(2));
        assert_eq!(5, data_index_to_node_index(3));
        assert_eq!(8, data_index_to_node_index(4));
        assert_eq!(9, data_index_to_node_index(5));
        assert_eq!(11, data_index_to_node_index(6));
        assert_eq!(12, data_index_to_node_index(7));
        assert_eq!(16, data_index_to_node_index(8));
        assert_eq!(17, data_index_to_node_index(9));
        assert_eq!(19, data_index_to_node_index(10));
        assert_eq!(20, data_index_to_node_index(11));
        assert_eq!(23, data_index_to_node_index(12));
        assert_eq!(24, data_index_to_node_index(13));
    }

    #[test]
    fn non_leaf_nodes_left_test() {
        assert_eq!(0, non_leaf_nodes_left(0));
        assert_eq!(0, non_leaf_nodes_left(1));
        assert_eq!(1, non_leaf_nodes_left(2));
        assert_eq!(1, non_leaf_nodes_left(3));
        assert_eq!(3, non_leaf_nodes_left(4));
        assert_eq!(3, non_leaf_nodes_left(5));
        assert_eq!(4, non_leaf_nodes_left(6));
        assert_eq!(4, non_leaf_nodes_left(7));
        assert_eq!(7, non_leaf_nodes_left(8));
        assert_eq!(7, non_leaf_nodes_left(9));
        assert_eq!(8, non_leaf_nodes_left(10));
        assert_eq!(8, non_leaf_nodes_left(11));
        assert_eq!(10, non_leaf_nodes_left(12));
        assert_eq!(10, non_leaf_nodes_left(13));
    }

    #[test]
    fn get_height_from_data_index_test() {
        assert_eq!(0, get_height_from_data_index(0));
        assert_eq!(1, get_height_from_data_index(1));
        assert_eq!(1, get_height_from_data_index(2));
        assert_eq!(2, get_height_from_data_index(3));
        assert_eq!(2, get_height_from_data_index(4));
        assert_eq!(2, get_height_from_data_index(5));
        assert_eq!(2, get_height_from_data_index(6));
        assert_eq!(3, get_height_from_data_index(7));
        assert_eq!(3, get_height_from_data_index(8));
    }

    #[test]
    fn data_index_node_index_pbt() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let rand = rng.next_u32();
            let inversion_result =
                node_index_to_data_index(data_index_to_node_index(rand as usize));
            match inversion_result {
                None => panic!(),
                Some(inversion) => assert_eq!(rand, inversion as u32),
            }
        }
    }

    #[test]
    fn is_right_child_test() {
        // Consider this a 1-indexed list of the expected result where the input to the function is the
        // (1-indexed) element of the list
        let anticipations: Vec<bool> = vec![
            false, true, false, false, true, true, false, false, true, false, false, true, true,
            //1      2     3      4      5     6     7      8      9     10     11     12    13
            true, false, false, true, false, false, true, true, false, false, true, false, false,
            //14     15    16     17    18     19    20    21     22    23     24     25    26
            true, true, true, true, false, false,
            //27    28   29    30    31     32
            true,
            //33
        ];

        for (i, anticipation) in anticipations.iter().enumerate() {
            assert!(right_child_and_height(i + 1).0 == *anticipation);
        }
    }

    #[test]
    fn leftmost_ancestor_test() {
        assert_eq!((1, 0), leftmost_ancestor(1));
        assert_eq!((3, 1), leftmost_ancestor(2));
        assert_eq!((3, 1), leftmost_ancestor(3));
        assert_eq!((7, 2), leftmost_ancestor(4));
        assert_eq!((7, 2), leftmost_ancestor(5));
        assert_eq!((7, 2), leftmost_ancestor(6));
        assert_eq!((7, 2), leftmost_ancestor(7));
        assert_eq!((15, 3), leftmost_ancestor(8));
        assert_eq!((15, 3), leftmost_ancestor(9));
        assert_eq!((15, 3), leftmost_ancestor(10));
        assert_eq!((15, 3), leftmost_ancestor(11));
        assert_eq!((15, 3), leftmost_ancestor(12));
        assert_eq!((15, 3), leftmost_ancestor(13));
        assert_eq!((15, 3), leftmost_ancestor(14));
        assert_eq!((15, 3), leftmost_ancestor(15));
        assert_eq!((31, 4), leftmost_ancestor(16));
    }

    #[test]
    fn left_sibling_test() {
        assert_eq!(3, left_sibling(6, 1));
        assert_eq!(1, left_sibling(2, 0));
        assert_eq!(4, left_sibling(5, 0));
        assert_eq!(15, left_sibling(30, 3));
        assert_eq!(22, left_sibling(29, 2));
        assert_eq!(7, left_sibling(14, 2));
    }

    #[test]
    fn node_index_to_data_index_test() {
        assert_eq!(Some(0), node_index_to_data_index(1));
        assert_eq!(Some(1), node_index_to_data_index(2));
        assert_eq!(None, node_index_to_data_index(3));
        assert_eq!(Some(2), node_index_to_data_index(4));
        assert_eq!(Some(3), node_index_to_data_index(5));
        assert_eq!(None, node_index_to_data_index(6));
        assert_eq!(None, node_index_to_data_index(7));
        assert_eq!(Some(4), node_index_to_data_index(8));
        assert_eq!(Some(5), node_index_to_data_index(9));
        assert_eq!(None, node_index_to_data_index(10));
        assert_eq!(Some(6), node_index_to_data_index(11));
        assert_eq!(Some(7), node_index_to_data_index(12));
        assert_eq!(None, node_index_to_data_index(13));
        assert_eq!(None, node_index_to_data_index(14));
        assert_eq!(None, node_index_to_data_index(15));
        assert_eq!(Some(8), node_index_to_data_index(16));
        assert_eq!(Some(9), node_index_to_data_index(17));
        assert_eq!(None, node_index_to_data_index(18));
        assert_eq!(Some(10), node_index_to_data_index(19));
        assert_eq!(Some(11), node_index_to_data_index(20));
        assert_eq!(None, node_index_to_data_index(21));
        assert_eq!(None, node_index_to_data_index(22));
    }

    #[test]
    fn one_input_mmr_test() {
        let element = vec![BFieldElement::new(14)];
        let mut rp = RescuePrimeProduction::new();
        let input_hash = rp.hash_one(&element);
        let mut mmr = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
            vec![input_hash.clone()],
            vec![BFieldElement::ring_zero()],
        );
        assert_eq!(1, mmr.count_leaves());
        assert_eq!(1, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, usize)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        assert_eq!(0, original_peaks_and_heights[0].1);
        let original_root: Vec<BFieldElement> = mmr.bag_peaks();

        let data_index = 0;
        let (authentication_path, peaks) = mmr.prove_membership(data_index);
        let valid = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
            &original_root,
            &authentication_path,
            &peaks,
            1,
            input_hash,
            data_index,
        );
        assert!(valid);

        let new_input_hash = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.archive_append(new_input_hash.clone());
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        assert_eq!(1, new_peaks_and_heights.len());
        assert_eq!(1, new_peaks_and_heights[0].1);

        let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
            .iter()
            .map(|x| x.0.to_vec())
            .collect();
        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        let new_root = mmr.bag_peaks();
        assert!(
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_input_hash,
                &new_peaks
            )
        );
    }

    #[test]
    fn two_input_mmr_test() {
        let values: Vec<Vec<BFieldElement>> = (0..2).map(|x| vec![BFieldElement::new(x)]).collect();
        let mut rp = RescuePrimeProduction::new();
        let input_hashes: Vec<Vec<BFieldElement>> = values.iter().map(|x| rp.hash_one(x)).collect();
        let mut mmr = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
            input_hashes.clone(),
            vec![BFieldElement::ring_zero()],
        );
        assert_eq!(2, mmr.count_leaves());
        assert_eq!(3, mmr.count_nodes());
        let original_peaks_and_heights: Vec<(Vec<BFieldElement>, usize)> =
            mmr.get_peaks_with_heights();
        assert_eq!(1, original_peaks_and_heights.len());
        let original_root = mmr.bag_peaks();
        let size = mmr.count_nodes() as u128;

        let data_index = 0;
        let (authentication_path, peaks) = mmr.prove_membership(data_index);
        let valid = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
            &original_root,
            &authentication_path,
            &peaks,
            size,
            input_hashes[data_index].clone(),
            data_index,
        );
        assert!(valid);

        let new_leaf_hash: Vec<BFieldElement> = rp.hash_one(&vec![BFieldElement::new(201)]);
        mmr.archive_append(new_leaf_hash.clone());
        let new_peaks_and_heights = mmr.get_peaks_with_heights();
        let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
            .iter()
            .map(|x| x.0.to_vec())
            .collect();
        let new_peaks: Vec<Vec<BFieldElement>> =
            new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
        let new_root = mmr.bag_peaks();
        assert!(
            ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_leaf_hash,
                &new_peaks
            )
        );
    }

    #[test]
    fn variable_size_rescue_prime_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<usize> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in izip!(
            (1usize..34).collect::<Vec<usize>>(),
            node_counts,
            peak_counts
        ) {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();
            let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            let input_hashes: Vec<Vec<BFieldElement>> =
                input_prehashes.iter().map(|x| rp.hash(x)).collect();
            let mut mmr = ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::init(
                input_hashes.clone(),
                vec![BFieldElement::ring_zero()],
            );
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights = mmr.get_peaks_with_heights();
            assert_eq!(peak_count, original_peaks_and_heights.len());
            let original_root = mmr.bag_peaks();
            let node_count = mmr.count_nodes();

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for index in 0..data_size {
                let (authentication_path, peaks) = mmr.prove_membership(index);
                let valid =
                    ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_membership(
                        &original_root,
                        &authentication_path,
                        &peaks,
                        node_count,
                        input_hashes[index].clone(),
                        index,
                    );
                assert!(valid);
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = rp.hash(&vec![BFieldElement::new(201)]);
            mmr.archive_append(new_leaf_hash.clone());
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<Vec<BFieldElement>> = original_peaks_and_heights
                .iter()
                .map(|x| x.0.to_vec())
                .collect();
            let new_peaks: Vec<Vec<BFieldElement>> =
                new_peaks_and_heights.iter().map(|x| x.0.to_vec()).collect();
            let new_root = mmr.bag_peaks();
            assert!(
                ArchivalMmr::<Vec<BFieldElement>, RescuePrimeProduction>::verify_append(
                    original_root,
                    &original_peaks,
                    mmr.count_leaves() - 1,
                    new_root,
                    new_leaf_hash,
                    &new_peaks
                )
            );
        }
    }

    #[test]
    fn variable_size_blake3_mmr_test() {
        let node_counts: Vec<u128> = vec![
            1, 3, 4, 7, 8, 10, 11, 15, 16, 18, 19, 22, 23, 25, 26, 31, 32, 34, 35, 38, 39, 41, 42,
            46, 47, 49, 50, 53, 54, 56, 57, 63, 64,
        ];
        let peak_counts: Vec<usize> = vec![
            1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4,
            4, 5, 1, 2,
        ];
        for (data_size, node_count, peak_count) in izip!(
            (1usize..34).collect::<Vec<usize>>(),
            node_counts,
            peak_counts
        ) {
            let input_prehashes: Vec<Vec<BFieldElement>> = (0..data_size)
                .map(|x| vec![BFieldElement::new(x as u128 + 14)])
                .collect();
            // let rp: RescuePrime = rescue_prime_params::rescue_prime_params_bfield_0();
            // blake3_digest(input)
            // blake3_digest_serialize()
            let input_hashes: Vec<blake3::Hash> = input_prehashes
                .iter()
                .map(|x| blake3::hash(bincode::serialize(x).expect("Encoding failed").as_slice()))
                .collect();
            let mut mmr = ArchivalMmr::<blake3::Hash, blake3::Hasher>::init(
                input_hashes.clone(),
                blake3::Hash::from_hex(format!("{:064x}", 0u128)).unwrap(),
            );
            assert_eq!(data_size, mmr.count_leaves());
            assert_eq!(node_count, mmr.count_nodes());
            let original_peaks_and_heights: Vec<(blake3::Hash, usize)> =
                mmr.get_peaks_with_heights();
            assert_eq!(peak_count, original_peaks_and_heights.len());
            let original_root = mmr.bag_peaks();
            let node_count = mmr.count_nodes();

            // Get an authentication path for **all** values in MMR,
            // verify that it is valid
            for index in 0..data_size {
                let (authentication_path, peaks) = mmr.prove_membership(index);
                let valid = ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_membership(
                    &original_root,
                    &authentication_path,
                    &peaks,
                    node_count,
                    input_hashes[index].clone(),
                    index,
                );
                assert!(valid);
            }

            // Make a new MMR where we append with a value and run the verify_append
            let new_leaf_hash = blake3::hash(
                blake3::Hash::from_hex(format!("{:064x}", 519u128))
                    .unwrap()
                    .as_bytes(),
            );
            mmr.archive_append(new_leaf_hash);
            let new_peaks_and_heights = mmr.get_peaks_with_heights();
            let original_peaks: Vec<blake3::Hash> =
                original_peaks_and_heights.iter().map(|x| x.0).collect();
            let new_peaks: Vec<blake3::Hash> = new_peaks_and_heights.iter().map(|x| x.0).collect();
            let new_root = mmr.bag_peaks();
            assert!(ArchivalMmr::<blake3::Hash, blake3::Hasher>::verify_append(
                original_root,
                &original_peaks,
                mmr.count_leaves() - 1,
                new_root,
                new_leaf_hash,
                &new_peaks
            ));
        }
    }
}
