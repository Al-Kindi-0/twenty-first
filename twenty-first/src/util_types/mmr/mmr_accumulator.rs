use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::{collections::HashMap, fmt::Debug};

use super::archival_mmr::ArchivalMmr;
use super::mmr_membership_proof::MmrMembershipProof;
use super::mmr_trait::Mmr;
use super::shared::{calculate_new_peaks_from_append, leaf_index_to_node_index};
use super::shared::{calculate_new_peaks_from_leaf_mutation, right_lineage_length_and_own_height};
use super::shared::{left_sibling, right_sibling};
use crate::shared_math::rescue_prime_digest::Digest;
use crate::util_types::algebraic_hasher::AlgebraicHasher;
use crate::util_types::mmr::shared::leaf_index_to_mt_index_and_peak_index;
use crate::util_types::shared::bag_peaks;
use crate::utils::has_unique_elements;

impl<H: AlgebraicHasher> From<ArchivalMmr<H>> for MmrAccumulator<H> {
    fn from(mut ammr: ArchivalMmr<H>) -> Self {
        MmrAccumulator {
            leaf_count: ammr.count_leaves(),
            peaks: ammr.get_peaks(),
            _hasher: PhantomData,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MmrAccumulator<H: AlgebraicHasher> {
    leaf_count: u128,
    peaks: Vec<Digest>,
    _hasher: PhantomData<H>,
}

impl<H: AlgebraicHasher> From<&mut ArchivalMmr<H>> for MmrAccumulator<H> {
    fn from(archive: &mut ArchivalMmr<H>) -> Self {
        Self {
            leaf_count: archive.count_leaves(),
            peaks: archive.get_peaks(),
            _hasher: PhantomData,
        }
    }
}

impl<H: AlgebraicHasher> MmrAccumulator<H> {
    pub fn init(peaks: Vec<Digest>, leaf_count: u128) -> Self {
        Self {
            leaf_count,
            peaks,
            _hasher: PhantomData,
        }
    }

    pub fn new(digests: Vec<Digest>) -> Self {
        let mut mmra = MmrAccumulator {
            leaf_count: 0,
            peaks: vec![],
            _hasher: PhantomData,
        };
        for digest in digests {
            mmra.append(digest);
        }

        mmra
    }
}

impl<H: AlgebraicHasher> Mmr<H> for MmrAccumulator<H> {
    fn bag_peaks(&mut self) -> Digest {
        bag_peaks::<H>(&self.peaks)
    }

    fn get_peaks(&mut self) -> Vec<Digest> {
        self.peaks.clone()
    }

    fn is_empty(&mut self) -> bool {
        self.leaf_count == 0
    }

    fn count_leaves(&mut self) -> u128 {
        self.leaf_count
    }

    fn append(&mut self, new_leaf: Digest) -> MmrMembershipProof<H> {
        let (new_peaks, membership_proof) =
            calculate_new_peaks_from_append::<H>(self.leaf_count, self.peaks.clone(), new_leaf)
                .unwrap();
        self.peaks = new_peaks;
        self.leaf_count += 1;

        membership_proof
    }

    /// Mutate an existing leaf. It is the caller's responsibility that the
    /// membership proof is valid. If the membership proof is wrong, the MMR
    /// will end up in a broken state.
    fn mutate_leaf(&mut self, old_membership_proof: &MmrMembershipProof<H>, new_leaf: &Digest) {
        self.peaks = calculate_new_peaks_from_leaf_mutation(
            &self.peaks,
            new_leaf,
            self.leaf_count,
            old_membership_proof,
        )
        .unwrap();
    }

    /// Returns true of the `new_peaks` input matches the calculated new MMR peaks resulting from the
    /// provided appends and mutations.
    fn verify_batch_update(
        &mut self,
        new_peaks: &[Digest],
        appended_leafs: &[Digest],
        leaf_mutations: &[(Digest, MmrMembershipProof<H>)],
    ) -> bool {
        // Verify that all leaf mutations operate on unique leafs and that they do
        // not exceed the total leaf count
        let manipulated_leaf_indices: Vec<u128> =
            leaf_mutations.iter().map(|x| x.1.leaf_index).collect();
        if !has_unique_elements(manipulated_leaf_indices.clone()) {
            return false;
        }

        // Disallow updating of out-of-bounds leafs
        if self.is_empty() && !manipulated_leaf_indices.is_empty()
            || !manipulated_leaf_indices.is_empty()
                && manipulated_leaf_indices.into_iter().max().unwrap() >= self.leaf_count
        {
            return false;
        }

        let mut leaf_mutation_target_values: Vec<Digest> =
            leaf_mutations.iter().map(|x| x.0.to_owned()).collect();
        let mut updated_membership_proofs: Vec<MmrMembershipProof<H>> =
            leaf_mutations.iter().map(|x| x.1.to_owned()).collect();

        // Reverse the leaf mutation vectors, since I would like to apply them in the order
        // they were input to this function using `pop`.
        leaf_mutation_target_values.reverse();
        updated_membership_proofs.reverse();

        // First we apply all the leaf mutations
        let mut running_peaks: Vec<Digest> = self.peaks.clone();
        while let Some(membership_proof) = updated_membership_proofs.pop() {
            // `new_leaf_value` is guaranteed to exist since `leaf_mutation_target_values`
            // has the same length as `updated_membership_proofs`
            let new_leaf_value = leaf_mutation_target_values.pop().unwrap();

            // TODO: Should we verify the membership proof here?

            // Calculate the new peaks after mutating a leaf
            let running_peaks_res = calculate_new_peaks_from_leaf_mutation(
                &running_peaks,
                &new_leaf_value,
                self.leaf_count,
                &membership_proof,
            );
            running_peaks = match running_peaks_res {
                None => return false,
                Some(peaks) => peaks,
            };

            // TODO: Replace this with the new batch updater
            // Update all remaining membership proofs with this leaf mutation
            MmrMembershipProof::<H>::batch_update_from_leaf_mutation(
                &mut updated_membership_proofs,
                &membership_proof,
                &new_leaf_value,
            );
        }

        // Then apply all the leaf appends
        let mut new_leafs_cloned: Vec<Digest> = appended_leafs.to_vec();

        // Reverse the new leafs to apply them in the same order as they were input,
        // using pop
        new_leafs_cloned.reverse();

        // Apply all leaf appends and
        let mut running_leaf_count = self.leaf_count;
        while let Some(new_leaf_for_append) = new_leafs_cloned.pop() {
            let append_res = calculate_new_peaks_from_append::<H>(
                running_leaf_count,
                running_peaks,
                new_leaf_for_append,
            );
            let (calculated_new_peaks, _new_membership_proof) = match append_res {
                None => return false,
                Some((peaks, mp)) => (peaks, mp),
            };
            running_peaks = calculated_new_peaks;
            running_leaf_count += 1;
        }

        running_peaks == new_peaks
    }

    fn batch_mutate_leaf_and_update_mps(
        &mut self,
        membership_proofs: &mut [&mut MmrMembershipProof<H>],
        mut mutation_data: Vec<(MmrMembershipProof<H>, Digest)>,
    ) -> Vec<usize> {
        // Calculate all derivable paths
        let mut new_ap_digests: HashMap<u128, Digest> = HashMap::new();

        // Calculate the derivable digests from a number of leaf mutations and their
        // associated authentication paths. Notice that all authentication paths
        // are only valid *prior* to any updates. They get invalidated (unless updated)
        // throughout the updating as their neighbor leaf digests change values.
        // The hash map `new_ap_digests` takes care of that.
        while let Some((ap, new_leaf)) = mutation_data.pop() {
            let mut node_index = leaf_index_to_node_index(ap.leaf_index);
            let former_value = new_ap_digests.insert(node_index, new_leaf);
            assert!(
                former_value.is_none(),
                "Duplicated leaf indices are not allowed in membership proof updater"
            );
            let mut acc_hash: Digest = new_leaf.to_owned();

            for (count, hash) in ap.authentication_path.iter().enumerate() {
                // If sibling node is something that has already been calculated, we use that
                // hash digest. Otherwise we use the one in our authentication path.
                let (right_ancestor_count, height) =
                    right_lineage_length_and_own_height(node_index);
                let is_right_child = right_ancestor_count != 0;
                if is_right_child {
                    let left_sibling_index = left_sibling(node_index, height);
                    let sibling_hash: &Digest = match new_ap_digests.get(&left_sibling_index) {
                        Some(h) => h,
                        None => hash,
                    };
                    acc_hash = H::hash_pair(sibling_hash, &acc_hash);

                    // Find parent node index
                    node_index += 1;
                } else {
                    let right_sibling_index = right_sibling(node_index, height);
                    let sibling_hash: &Digest = match new_ap_digests.get(&right_sibling_index) {
                        Some(h) => h,
                        None => hash,
                    };
                    acc_hash = H::hash_pair(&acc_hash, sibling_hash);

                    // Find parent node index
                    node_index += 1 << (height + 1);
                }

                // The last hash calculated is the peak hash
                // This is not inserted in the hash map, as it will never be in any
                // authentication path
                if count < ap.authentication_path.len() - 1 {
                    new_ap_digests.insert(node_index, acc_hash);
                }
            }

            // Update the peak
            let (_, peak_index) =
                leaf_index_to_mt_index_and_peak_index(ap.leaf_index, self.count_leaves());
            self.peaks[peak_index as usize] = acc_hash;
        }

        // Update all the supplied membership proofs
        let mut modified_membership_proof_indices: Vec<usize> = vec![];
        for (i, membership_proof) in membership_proofs.iter_mut().enumerate() {
            let ap_indices = membership_proof.get_node_indices();

            // Some of the hashes in may `membership_proof` need to be updated. We can loop over
            // `authentication_path_indices` and check if the element is contained `deducible_hashes`.
            // If it is, then the appropriate element in `membership_proof.authentication_path` needs to
            // be replaced with an element from `deducible_hashes`.
            for (digest, authentication_path_indices) in membership_proof
                .authentication_path
                .iter_mut()
                .zip(ap_indices.into_iter())
            {
                // Any number of hashes can be updated in the authentication path, since
                // we're modifying multiple leaves in the MMR
                // Since this function returns the indices of the modified membership proofs,
                // a check if the new digest is actually different from the previous value is
                // needed.
                if new_ap_digests.contains_key(&authentication_path_indices)
                    && *digest != new_ap_digests[&authentication_path_indices]
                {
                    *digest = new_ap_digests[&authentication_path_indices];
                    modified_membership_proof_indices.push(i);
                }
            }
        }

        modified_membership_proof_indices.dedup();
        modified_membership_proof_indices
    }

    fn to_accumulator(&mut self) -> MmrAccumulator<H> {
        self.to_owned()
    }
}

#[cfg(test)]
mod accumulator_mmr_tests {
    use std::cmp;

    use itertools::{izip, Itertools};
    use num_traits::Zero;
    use rand::Rng;

    use crate::shared_math::b_field_element::BFieldElement;
    use crate::shared_math::other::{random_elements, random_elements_range};
    use crate::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use crate::test_shared::mmr::get_archival_mmr_from_digests;

    use super::*;

    #[test]
    fn conversion_test() {
        type H = blake3::Hasher;

        let leaf_hashes: Vec<Digest> = random_elements(3);
        let mut archival_mmr: ArchivalMmr<H> = get_archival_mmr_from_digests(leaf_hashes);
        let mut accumulator_mmr: MmrAccumulator<H> = (&mut archival_mmr).into();

        assert_eq!(archival_mmr.get_peaks(), accumulator_mmr.get_peaks());
        assert_eq!(archival_mmr.bag_peaks(), accumulator_mmr.bag_peaks());
        assert_eq!(archival_mmr.is_empty(), accumulator_mmr.is_empty());
        assert!(!archival_mmr.is_empty());
        assert_eq!(archival_mmr.count_leaves(), accumulator_mmr.count_leaves());
        assert_eq!(3, accumulator_mmr.count_leaves());
    }

    #[test]
    fn verify_batch_update_single_append_test() {
        type H = blake3::Hasher;

        let leaf_hashes_start: Vec<Digest> = random_elements(3);
        let appended_leaf: Digest = H::hash(&17u128);

        let mut leaf_hashes_end: Vec<Digest> = leaf_hashes_start.clone();
        leaf_hashes_end.push(appended_leaf);

        let mut accumulator_mmr_start: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_start);
        let mut accumulator_mmr_end: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_end);

        let leaves_were_appended = accumulator_mmr_start.verify_batch_update(
            &accumulator_mmr_end.get_peaks(),
            &[appended_leaf],
            &[],
        );
        assert!(leaves_were_appended);
    }

    #[test]
    fn verify_batch_update_single_mutate_test() {
        type H = blake3::Hasher;

        let leaf_hashes_start: Vec<Digest> = [14, 15, 16, 18].iter().map(H::hash::<u128>).collect();
        let new_leaf_value: Digest = H::hash(&17u128);

        let leaf_hashes_end: Vec<Digest> = [14, 15, 16, 17].iter().map(H::hash::<u128>).collect();

        let mut accumulator_mmr_start: MmrAccumulator<H> =
            MmrAccumulator::new(leaf_hashes_start.clone());
        let mut archive_mmr_start: ArchivalMmr<H> =
            get_archival_mmr_from_digests(leaf_hashes_start);
        let membership_proof = archive_mmr_start.prove_membership(3).0;
        let mut accumulator_mmr_end: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_end);

        {
            let appended_leafs = [];
            let leaf_mutations = [(new_leaf_value, membership_proof.clone())];
            assert!(accumulator_mmr_start.verify_batch_update(
                &accumulator_mmr_end.get_peaks(),
                &appended_leafs,
                &leaf_mutations,
            ));
        }
        // Verify that repeated indices are disallowed
        {
            let appended_leafs = [];
            let leaf_mutations = [
                (new_leaf_value, membership_proof.clone()),
                (new_leaf_value, membership_proof),
            ];
            assert!(!accumulator_mmr_start.verify_batch_update(
                &accumulator_mmr_end.get_peaks(),
                &appended_leafs,
                &leaf_mutations,
            ));
        }
    }

    #[test]
    fn verify_batch_update_two_append_test() {
        type H = blake3::Hasher;

        let leaf_hashes_start: Vec<Digest> = random_elements(3);
        let appended_leafs: Vec<Digest> = random_elements(2);
        let leaf_hashes_end: Vec<Digest> =
            vec![leaf_hashes_start.clone(), appended_leafs.clone()].concat();
        let mut accumulator_mmr_start: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_start);
        let mut accumulator_mmr_end: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_end);

        let leaves_were_appended = accumulator_mmr_start.verify_batch_update(
            &accumulator_mmr_end.get_peaks(),
            &appended_leafs,
            &[],
        );
        assert!(leaves_were_appended);
    }

    #[test]
    fn verify_batch_update_two_mutate_test() {
        type H = blake3::Hasher;

        let leaf_hashes_start: Vec<Digest> = [14, 15, 16, 17].iter().map(H::hash::<u128>).collect();
        let new_leafs: Vec<Digest> = [20, 21].iter().map(H::hash::<u128>).collect();
        let leaf_hashes_end: Vec<Digest> = [14, 20, 16, 21].iter().map(H::hash::<u128>).collect();

        let mut accumulator_mmr_start: MmrAccumulator<H> =
            MmrAccumulator::<H>::new(leaf_hashes_start.clone());
        let mut archive_mmr_start: ArchivalMmr<H> =
            get_archival_mmr_from_digests(leaf_hashes_start);
        let membership_proof1 = archive_mmr_start.prove_membership(1).0;
        let membership_proof3 = archive_mmr_start.prove_membership(3).0;
        let mut accumulator_mmr_end: MmrAccumulator<H> = MmrAccumulator::new(leaf_hashes_end);
        assert!(accumulator_mmr_start.verify_batch_update(
            &accumulator_mmr_end.get_peaks(),
            &[],
            &[
                (new_leafs[0], membership_proof1),
                (new_leafs[1], membership_proof3)
            ]
        ));
    }

    #[test]
    fn batch_mutate_leaf_and_update_mps_test() {
        type H = blake3::Hasher;

        let mut rng = rand::thread_rng();
        for mmr_leaf_count in 1..100 {
            let initial_leaf_digests: Vec<Digest> = random_elements(mmr_leaf_count);

            let mut mmra: MmrAccumulator<H> = MmrAccumulator::new(initial_leaf_digests.clone());
            let mut ammr: ArchivalMmr<H> =
                get_archival_mmr_from_digests(initial_leaf_digests.clone());
            let mut ammr_copy: ArchivalMmr<H> =
                get_archival_mmr_from_digests(initial_leaf_digests.clone());

            let mutated_leaf_count = rng.gen_range(0..mmr_leaf_count);
            let all_indices: Vec<u128> = (0..mmr_leaf_count as u128).collect();

            // Pick indices for leaves that are being mutated
            let mut all_indices_mut0 = all_indices.clone();
            let mut mutated_leaf_indices: Vec<u128> = vec![];
            for _ in 0..mutated_leaf_count {
                let leaf_index = all_indices_mut0.remove(rng.gen_range(0..all_indices_mut0.len()));
                mutated_leaf_indices.push(leaf_index);
            }

            // Pick membership proofs that we want to update
            let membership_proof_count = rng.gen_range(0..mmr_leaf_count);
            let mut all_indices_mut1 = all_indices.clone();
            let mut membership_proof_indices: Vec<u128> = vec![];
            for _ in 0..membership_proof_count {
                let leaf_index = all_indices_mut1.remove(rng.gen_range(0..all_indices_mut1.len()));
                membership_proof_indices.push(leaf_index);
            }

            // Calculate the terminal leafs, as they look after the batch leaf mutation
            // that we are preparing to execute
            let new_leafs: Vec<Digest> = random_elements(mutated_leaf_count);
            let mut terminal_leafs: Vec<Digest> = initial_leaf_digests;
            for (i, new_leaf) in mutated_leaf_indices.iter().zip(new_leafs.iter()) {
                terminal_leafs[*i as usize] = new_leaf.to_owned();
            }

            // Calculate the leafs digests associated with the membership proofs, as they look
            // *after* the batch leaf mutation
            let mut terminal_leafs_for_mps: Vec<Digest> = vec![];
            for i in membership_proof_indices.iter() {
                terminal_leafs_for_mps.push(terminal_leafs[*i as usize]);
            }

            // Construct the mutation data
            let mutated_leaf_mps = mutated_leaf_indices
                .iter()
                .map(|i| ammr.prove_membership(*i).0);
            let mutation_data: Vec<(MmrMembershipProof<H>, Digest)> = mutated_leaf_mps
                .into_iter()
                .zip(new_leafs.into_iter())
                .collect();

            assert_eq!(mutated_leaf_count, mutation_data.len());

            let original_membership_proofs: Vec<MmrMembershipProof<H>> = membership_proof_indices
                .iter()
                .map(|i| ammr.prove_membership(*i).0)
                .collect();

            // Do the update on both MMRs
            let mut mmra_mps = original_membership_proofs.clone();
            let mut ammr_mps = original_membership_proofs.clone();
            let mutated_mps_mmra = mmra.batch_mutate_leaf_and_update_mps(
                &mut mmra_mps.iter_mut().collect::<Vec<_>>(),
                mutation_data.clone(),
            );
            let mutated_mps_ammr = ammr.batch_mutate_leaf_and_update_mps(
                &mut ammr_mps.iter_mut().collect::<Vec<_>>(),
                mutation_data.clone(),
            );
            assert_eq!(mutated_mps_mmra, mutated_mps_ammr);

            // Verify that both MMRs end up with same peaks
            assert_eq!(mmra.get_peaks(), ammr.get_peaks());

            // Verify that membership proofs from AMMR and MMRA are equal
            assert_eq!(membership_proof_count, mmra_mps.len());
            assert_eq!(membership_proof_count, ammr_mps.len());
            assert_eq!(ammr_mps, mmra_mps);

            // Verify that all membership proofs still work
            assert!(mmra_mps
                .iter()
                .zip(terminal_leafs_for_mps.iter())
                .all(|(mp, leaf)| mp.verify(&mmra.get_peaks(), leaf, mmra.count_leaves()).0));

            // Manually construct an MMRA from the new data and verify that peaks and leaf count matches
            assert!(
                mutated_leaf_count == 0 || ammr_copy.get_peaks() != ammr.get_peaks(),
                "If mutated leaf count is non-zero, at least on peaks must be different"
            );
            mutation_data.into_iter().for_each(|(mp, digest)| {
                ammr_copy.mutate_leaf_raw(mp.leaf_index, digest);
            });
            assert_eq!(ammr_copy.get_peaks(), ammr.get_peaks(), "Mutation though batch mutation function must transform the MMR like a list of individual leaf mutations");
        }
    }

    #[test]
    fn verify_batch_update_pbt() {
        type H = blake3::Hasher;

        for start_size in 1..35 {
            let leaf_hashes_start: Vec<Digest> = random_elements(start_size);

            let bad_digests: Vec<Digest> = (12..12 + start_size)
                .map(|x| H::hash(&(x as u128)))
                .collect();

            let mut bad_mmr: ArchivalMmr<H> = get_archival_mmr_from_digests(bad_digests.clone());
            let bad_membership_proof: MmrMembershipProof<H> = bad_mmr.prove_membership(0).0;
            let bad_membership_proof_digest = bad_digests[0];
            let bad_leaf: Digest = H::hash(&8765432165123u128);
            let mut archival_mmr_init: ArchivalMmr<H> =
                get_archival_mmr_from_digests(leaf_hashes_start.clone());
            let mut accumulator_mmr = MmrAccumulator::<H>::new(leaf_hashes_start.clone());

            for append_size in 0..18 {
                let appends: Vec<Digest> =
                    (2000..2000 + append_size).map(|x| H::hash(&x)).collect();
                let mutate_count = cmp::min(12, start_size);
                for mutate_size in 0..mutate_count {
                    let new_leaf_values: Vec<Digest> = (13..13 + mutate_size)
                        .map(|x| H::hash(&(x as u128)))
                        .collect();

                    // Ensure that indices are unique since batch updating cannot update
                    // the same leaf twice in one go
                    let mutated_indices: Vec<u128> =
                        random_elements_range(mutate_size, 0..start_size as u128)
                            .into_iter()
                            .sorted()
                            .unique()
                            .collect();

                    // Create the expected MMRs
                    let mut leaf_hashes_mutated = leaf_hashes_start.clone();
                    for (index, new_leaf) in izip!(mutated_indices.clone(), new_leaf_values.clone())
                    {
                        leaf_hashes_mutated[index as usize] = new_leaf;
                    }
                    for appended_digest in appends.iter() {
                        leaf_hashes_mutated.push(appended_digest.to_owned());
                    }

                    // let mutated_archival_mmr =
                    //     ArchivalMmr::<Hasher>::new(leaf_hashes_mutated.clone());
                    let mut mutated_archival_mmr: ArchivalMmr<H> =
                        get_archival_mmr_from_digests(leaf_hashes_mutated.clone());
                    let mut mutated_accumulator_mmr = MmrAccumulator::<H>::new(leaf_hashes_mutated);
                    let expected_new_peaks_from_archival = mutated_archival_mmr.get_peaks();
                    let expected_new_peaks_from_accumulator = mutated_accumulator_mmr.get_peaks();
                    assert_eq!(
                        expected_new_peaks_from_archival,
                        expected_new_peaks_from_accumulator
                    );

                    // Create the inputs to the method call
                    let membership_proofs = mutated_indices
                        .iter()
                        .map(|&i| archival_mmr_init.prove_membership(i).0);
                    let mut leaf_mutations: Vec<(Digest, MmrMembershipProof<H>)> = new_leaf_values
                        .clone()
                        .into_iter()
                        .zip(membership_proofs)
                        .map(|(v, mp)| (v, mp))
                        .collect();
                    assert!(accumulator_mmr.verify_batch_update(
                        &expected_new_peaks_from_accumulator,
                        &appends,
                        &leaf_mutations
                    ));
                    assert!(archival_mmr_init.verify_batch_update(
                        &expected_new_peaks_from_accumulator,
                        &appends,
                        &leaf_mutations
                    ));

                    // Negative tests
                    let mut bad_appends = appends.clone();
                    if append_size > 0 && mutate_size > 0 {
                        // bad append vector
                        bad_appends[(mutated_indices[0] % append_size) as usize] = bad_leaf;
                        assert!(!accumulator_mmr.verify_batch_update(
                            &expected_new_peaks_from_accumulator,
                            &bad_appends,
                            &leaf_mutations
                        ));

                        // Bad membership proof
                        let bad_index = mutated_indices[0] as usize % mutated_indices.len();
                        leaf_mutations[bad_index].0 = bad_membership_proof_digest;
                        assert!(!accumulator_mmr.verify_batch_update(
                            &expected_new_peaks_from_accumulator,
                            &appends,
                            &leaf_mutations
                        ));
                        leaf_mutations[mutated_indices[0] as usize % mutated_indices.len()].1 =
                            bad_membership_proof.clone();
                        assert!(!accumulator_mmr.verify_batch_update(
                            &expected_new_peaks_from_accumulator,
                            &appends,
                            &leaf_mutations
                        ));
                    }
                }
            }
        }
    }

    #[test]
    fn mmra_serialization_test() {
        // You could argue that this test doesn't belong here, as it tests the behavior of
        // an imported library. I included it here, though, because the setup seems a bit clumsy
        // to me so far.
        type H = RescuePrimeRegular;
        type Mmr = MmrAccumulator<H>;
        let mut mmra: Mmr = MmrAccumulator::new(vec![]);
        mmra.append(H::hash(&BFieldElement::zero()));

        let json = serde_json::to_string(&mmra).unwrap();
        let mut s_back = serde_json::from_str::<Mmr>(&json).unwrap();
        assert!(mmra.bag_peaks() == s_back.bag_peaks());
        assert_eq!(1, mmra.count_leaves());
    }
}
