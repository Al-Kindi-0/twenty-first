use std::collections::HashMap;
use std::error::Error;

use super::super::triton;
use super::table::base_matrix::BaseMatrices;
use super::vm::Program;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::other;
use crate::shared_math::other::roundup_npo2;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::rescue_prime_xlix::{
    neptune_params, RescuePrimeXlix, RP_DEFAULT_OUTPUT_SIZE, RP_DEFAULT_WIDTH,
};
use crate::shared_math::stark::stark_verify_error::StarkVerifyError;
use crate::shared_math::stark::triton::arguments::evaluation_argument::all_evaluation_arguments;
use crate::shared_math::stark::triton::arguments::permutation_argument::PermArg;
use crate::shared_math::stark::triton::instruction::sample_programs;
use crate::shared_math::stark::triton::proof_item::{Item, StarkProofStream};
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::stark::triton::table::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::stark::triton::table::table_collection::{
    BaseTableCollection, ExtTableCollection,
};
use crate::shared_math::stark::triton::triton_xfri;
use crate::shared_math::traits::{GetPrimitiveRootOfUnity, GetRandomElements, Inverse, ModPowU32};
use crate::shared_math::x_field_element::XFieldElement;
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::{MerkleTree, PartialAuthenticationPath};
use crate::util_types::proof_stream_typed::ProofStream;
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

type BWord = BFieldElement;
type XWord = XFieldElement;
type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;
type StarkDigest = Vec<BFieldElement>;

// We use a type-parameterised FriDomain to avoid duplicate `b_*()` and `x_*()` methods.
pub struct Stark {
    _padded_height: usize,
    _log_expansion_factor: usize,
    num_randomizers: usize,
    order: usize,
    generator: XFieldElement,
    security_level: usize,
    max_degree: Degree,
    bfri_domain: triton::fri_domain::FriDomain<BWord>,
    xfri_domain: triton::fri_domain::FriDomain<XWord>,
    fri: triton_xfri::Fri<StarkHasher>,
}

impl Stark {
    pub fn new(_padded_height: usize, log_expansion_factor: usize, security_level: usize) -> Self {
        assert_eq!(
            0,
            security_level % log_expansion_factor,
            "security_level/log_expansion_factor must be a positive integer"
        );

        let expansion_factor: u64 = 1 << log_expansion_factor;
        let colinearity_checks: usize = security_level / log_expansion_factor;

        assert!(
            colinearity_checks > 0,
            "At least one colinearity check is required"
        );

        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let num_randomizers = 2;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();

        let code = sample_programs::HELLO_WORLD_1;
        let program = Program::from_code(code).unwrap();

        let (base_matrices, _err) = program.simulate_with_input(&[], &[]);

        let base_table_collection = BaseTableCollection::from_base_matrices(
            smooth_generator,
            order,
            num_randomizers,
            &base_matrices,
        );

        let max_degree =
            (other::roundup_npo2(base_table_collection.max_degree() as u64) - 1) as i64;
        let fri_domain_length = ((max_degree as u64 + 1) * expansion_factor) as usize;

        let offset = BWord::generator();
        let omega = BWord::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u64)
            .0
            .unwrap();

        let bfri_domain = triton::fri_domain::FriDomain {
            offset,
            omega,
            length: fri_domain_length as usize,
        };

        let dummy_xfri_domain = triton::fri_domain::FriDomain::<XFieldElement> {
            offset: offset.lift(),
            omega: omega.lift(),
            length: fri_domain_length as usize,
        };

        let dummy_xfri = triton_xfri::Fri::new(
            offset.lift(),
            omega.lift(),
            fri_domain_length,
            expansion_factor as usize,
            colinearity_checks,
        );

        Stark {
            _padded_height,
            _log_expansion_factor: log_expansion_factor,
            num_randomizers,
            order,
            generator: smooth_generator.lift(),
            security_level,
            max_degree,
            bfri_domain,
            xfri_domain: dummy_xfri_domain,
            fri: dummy_xfri,
        }
    }

    pub fn prove(&self, base_matrices: BaseMatrices) -> StarkProofStream {
        let mut timer = TimingReporter::start();

        let num_randomizers = 1;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();
        let unpadded_height = base_matrices.processor_matrix.len();
        let _padded_height = roundup_npo2(unpadded_height as u64);

        // 1. Create base tables based on base matrices

        let mut base_tables = BaseTableCollection::from_base_matrices(
            smooth_generator,
            order,
            num_randomizers,
            &base_matrices,
        );

        timer.elapsed("assert, set_matrices");

        base_tables.pad();

        timer.elapsed("pad");

        let max_degree = base_tables.max_degree();

        // Randomizer bla bla
        let mut rng = rand::thread_rng();
        let randomizer_coefficients =
            XFieldElement::random_elements(max_degree as usize + 1, &mut rng);
        let randomizer_polynomial = Polynomial::new(randomizer_coefficients);

        let x_randomizer_codeword: Vec<XFieldElement> =
            self.fri.domain.x_evaluate(&randomizer_polynomial);
        let mut b_randomizer_codewords: [Vec<BFieldElement>; 3] = [vec![], vec![], vec![]];
        for x_elem in x_randomizer_codeword.iter() {
            b_randomizer_codewords[0].push(x_elem.coefficients[0]);
            b_randomizer_codewords[1].push(x_elem.coefficients[1]);
            b_randomizer_codewords[2].push(x_elem.coefficients[2]);
        }

        timer.elapsed("randomizer_codewords");

        let base_codewords: Vec<Vec<BFieldElement>> =
            base_tables.all_base_codewords(&self.bfri_domain);

        let all_base_codewords =
            vec![b_randomizer_codewords.into(), base_codewords.clone()].concat();

        timer.elapsed("get_and_set_all_base_codewords");

        let transposed_base_codewords: Vec<Vec<BFieldElement>> = (0..all_base_codewords[0].len())
            .map(|i| {
                all_base_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<BFieldElement>>()
            })
            .collect();

        timer.elapsed("transposed_base_codewords");

        let hasher = neptune_params();

        let mut base_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_base_codewords.len());

        transposed_base_codewords
            .par_iter()
            .map(|values| hasher.hash(values, DIGEST_LEN))
            .collect_into_vec(&mut base_codeword_digests_by_index);

        let base_merkle_tree =
            MerkleTree::<StarkHasher>::from_digests(&base_codeword_digests_by_index);

        timer.elapsed("base_merkle_tree");

        // Commit to base codewords

        let mut proof_stream: ProofStream<Item, StarkHasher> = StarkProofStream::default();
        let base_merkle_tree_root: Vec<BFieldElement> = base_merkle_tree.get_root();
        proof_stream.enqueue(&Item::MerkleRoot(base_merkle_tree_root));

        timer.elapsed("proof_stream.enqueue");

        let seed = proof_stream.prover_fiat_shamir();

        timer.elapsed("prover_fiat_shamir");

        let challenge_weights =
            Self::sample_weights(&hasher, &seed, AllChallenges::TOTAL_CHALLENGES);
        let all_challenges: AllChallenges = AllChallenges::create_challenges(&challenge_weights);

        timer.elapsed("sample_weights");

        let initial_weights = Self::sample_weights(&hasher, &seed, AllEndpoints::TOTAL_ENDPOINTS);
        let all_initials: AllEndpoints = AllEndpoints::create_initials(&initial_weights);

        timer.elapsed("initials");

        let (ext_tables, all_terminals) =
            ExtTableCollection::extend_tables(&base_tables, &all_challenges, &all_initials);
        let ext_codeword_tables = ext_tables.codeword_tables(&self.xfri_domain);
        let all_ext_codewords: Vec<Vec<XWord>> = ext_codeword_tables.concat_table_data();

        timer.elapsed("extend + get_terminals");

        timer.elapsed("get_and_set_all_extension_codewords");

        let transposed_extension_codewords: Vec<Vec<XFieldElement>> = (0..all_ext_codewords[0]
            .len())
            .map(|i| {
                all_ext_codewords
                    .iter()
                    .map(|inner| inner[i])
                    .collect::<Vec<XFieldElement>>()
            })
            .collect();

        let mut extension_codeword_digests_by_index: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(transposed_extension_codewords.len());

        let transposed_extension_codewords_clone = transposed_extension_codewords.clone();
        transposed_extension_codewords_clone
            .into_par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .into_iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();
                assert_eq!(
                    27,
                    bvalues.len(),
                    "9 X-field elements must become 27 B-field elements"
                );
                hasher.hash(&bvalues, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut extension_codeword_digests_by_index);

        let extension_tree =
            MerkleTree::<StarkHasher>::from_digests(&extension_codeword_digests_by_index);
        proof_stream.enqueue(&Item::MerkleRoot(extension_tree.get_root()));
        proof_stream.enqueue(&Item::Terminals(all_terminals.clone()));

        timer.elapsed("extension_tree");

        let extension_degree_bounds: Vec<Degree> = ext_tables.get_all_extension_degree_bounds();

        timer.elapsed("get_all_extension_degree_bounds");

        let mut quotient_codewords =
            ext_tables.get_all_quotients(&self.bfri_domain, &all_challenges, &all_terminals);

        timer.elapsed("all_quotients");

        let mut quotient_degree_bounds =
            ext_tables.get_all_quotient_degree_bounds(&all_challenges, &all_terminals);

        timer.elapsed("all_quotient_degree_bounds");

        // Prove equal initial values for the permutation-extension column pairs
        for pa in PermArg::all_permutation_arguments().iter() {
            quotient_codewords.push(pa.quotient(&ext_codeword_tables, &self.fri.domain));
            quotient_degree_bounds.push(pa.quotient_degree_bound(&ext_codeword_tables));
        }

        // Calculate `num_base_polynomials` and `num_extension_polynomials` for asserting
        let num_base_polynomials: usize = base_tables.into_iter().map(|table| table.width()).sum();
        let num_extension_polynomials: usize = ext_tables
            .into_iter()
            .map(|ext_table| ext_table.width() - ext_table.base_width())
            .sum();

        timer.elapsed("num_(base+extension)_polynomials");

        let num_randomizer_polynomials: usize = 1;
        let num_quotient_polynomials: usize = quotient_degree_bounds.len();
        let base_degree_bounds = base_tables.get_all_base_degree_bounds();

        timer.elapsed("get_all_base_degree_bounds");

        // Get weights for nonlinear combination
        let weights_seed: Vec<BFieldElement> = proof_stream.prover_fiat_shamir();

        timer.elapsed("prover_fiat_shamir (again)");

        let weights_count = num_randomizer_polynomials
            + 2 * (num_base_polynomials + num_extension_polynomials + num_quotient_polynomials);
        let weights = Self::sample_weights(&hasher, &weights_seed, weights_count);

        timer.elapsed("sample_weights");

        // let mut terms: Vec<Vec<XFieldElement>> = vec![x_randomizer_codeword];
        let mut weights_counter = 0;
        let mut combination_codeword: Vec<XFieldElement> = x_randomizer_codeword
            .into_iter()
            .map(|elem| elem * weights[weights_counter])
            .collect();
        weights_counter += 1;
        assert_eq!(base_codewords.len(), num_base_polynomials);
        let fri_x_values: Vec<BFieldElement> = self.fri.domain.b_domain_values();

        timer.elapsed("b_domain_values");

        for (i, (bc, bdb)) in base_codewords
            .iter()
            .zip(base_degree_bounds.iter())
            .enumerate()
        {
            let bc_lifted: Vec<XFieldElement> = bc.iter().map(|bfe| bfe.lift()).collect();

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(bc_lifted.into_par_iter())
                .map(|(c, bcl)| c + bcl * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - bdb) as u32;
            let bc_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(bc.par_iter())
                .map(|(x, &bce)| (bce * x.mod_pow_u32(shift)).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&bc_shifted);
                assert!(
                    interpolated.degree() == -1 || interpolated.degree() == max_degree as isize,
                    "The shifted base codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    max_degree,
                    interpolated.degree()
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(bc_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }

        timer.elapsed("...shift and collect base codewords");

        assert_eq!(all_ext_codewords.len(), num_extension_polynomials);
        for (i, (ec, edb)) in all_ext_codewords
            .iter()
            .zip(extension_degree_bounds.iter())
            .enumerate()
        {
            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(ec.par_iter())
                .map(|(c, new_elem)| c + *new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - edb) as u32;
            let ec_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(ec.into_par_iter())
                .map(|(x, &ece)| ece * x.mod_pow_u32(shift).lift())
                .collect();

            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&ec_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == max_degree as isize,
                    "The shifted extension codeword with index {} must be of maximal degree {}. Got {}.",
                    i,
                    max_degree,
                    interpolated.degree()
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(ec_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }
        timer.elapsed("...shift and collect extension codewords");

        assert_eq!(quotient_codewords.len(), num_quotient_polynomials);
        for (_i, (qc, qdb)) in quotient_codewords
            .iter()
            .zip(quotient_degree_bounds.iter())
            .enumerate()
        {
            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(qc.par_iter())
                .map(|(c, new_elem)| c + *new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
            let shift = (max_degree as Degree - qdb) as u32;
            let qc_shifted: Vec<XFieldElement> = fri_x_values
                .par_iter()
                .zip(qc.into_par_iter())
                .map(|(x, &qce)| qce * x.mod_pow_u32(shift).lift())
                .collect();

            // TODO: Not all the degrees of the shifted quotient codewords are of max degree. Why?
            if std::env::var("DEBUG").is_ok() {
                let interpolated = self.fri.domain.x_interpolate(&qc_shifted);
                assert!(
                    interpolated.degree() == -1
                        || interpolated.degree() == max_degree as isize,
                    "The shifted quotient codeword with index {} must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: {}. . Shift = {}",
                    _i,
                    max_degree,
                    interpolated.degree(),
                    qdb,
                    shift
                );
            }

            combination_codeword = combination_codeword
                .into_par_iter()
                .zip(qc_shifted.into_par_iter())
                .map(|(c, new_elem)| c + new_elem * weights[weights_counter])
                .collect();
            weights_counter += 1;
        }

        timer.elapsed("...shift and collect quotient codewords");

        let mut combination_codeword_digests: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(combination_codeword.len());
        combination_codeword
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let digest: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&digest, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut combination_codeword_digests);
        let combination_tree =
            MerkleTree::<StarkHasher>::from_digests(&combination_codeword_digests);
        let combination_root: Vec<BFieldElement> = combination_tree.get_root();
        proof_stream.enqueue(&Item::MerkleRoot(combination_root.clone()));

        timer.elapsed("combination_tree");

        // TODO: Consider factoring out code to find `unit_distances`, duplicated in verifier
        let mut unit_distances: Vec<usize> = ext_tables // XXX:
            .into_iter()
            .map(|table| table.unit_distance(self.fri.domain.length))
            .collect();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();

        timer.elapsed("unit_distances");

        // Get indices of leafs to prove nonlinear combination
        let indices_seed: Vec<BFieldElement> = proof_stream.prover_fiat_shamir();
        let indices: Vec<usize> =
            hasher.sample_indices(self.security_level, &indices_seed, self.fri.domain.length);

        timer.elapsed("sample_indices");

        // TODO: I don't like that we're calling FRI right after getting the indices through
        // the Fiat-Shamir public oracle above. The reason I don't like this is that it implies
        // using Fiat-Shamir twice with somewhat similar proof stream content. A cryptographer
        // or mathematician should take a look on this part of the code.
        // prove low degree of combination polynomial
        let (_fri_indices, combination_root_verify) = self
            .fri
            .prove(&combination_codeword, &mut proof_stream)
            .unwrap();
        timer.elapsed("fri.prove");
        assert_eq!(
            combination_root, combination_root_verify,
            "Combination root from STARK and from FRI must agree"
        );

        // Open leafs of zipped codewords at indicated positions
        let mut revealed_indices: Vec<usize> = vec![];
        for index in indices.iter() {
            for unit_distance in unit_distances.iter() {
                let idx: usize = (index + unit_distance) % self.fri.domain.length;
                revealed_indices.push(idx);
            }
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();

        let revealed_elements: Vec<Vec<BFieldElement>> = revealed_indices
            .iter()
            .map(|idx| transposed_base_codewords[*idx].clone())
            .collect();
        let auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> =
            base_merkle_tree.get_multi_proof(&revealed_indices);

        proof_stream.enqueue(&Item::TransposedBaseElementVectors(revealed_elements));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(auth_paths));

        let revealed_extension_elements: Vec<Vec<XFieldElement>> = revealed_indices
            .iter()
            .map(|idx| transposed_extension_codewords[*idx].clone())
            .collect_vec();
        let extension_auth_paths = extension_tree.get_multi_proof(&revealed_indices);
        proof_stream.enqueue(&Item::TransposedExtensionElementVectors(
            revealed_extension_elements,
        ));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(extension_auth_paths));

        // debug_assert!(
        //     MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
        //         base_merkle_tree.get_root(),
        //         &revealed_indices,
        //         revealed_indices.iter().map()base_codeword_digests_by_index[idx].clone(),
        //         auth_path.clone(),
        //     ),
        //     "authentication path for base tree must be valid"
        // );

        timer.elapsed("open leafs of zipped codewords");

        // open combination codeword at the same positions
        // Notice that we need to loop over `indices` here, not `revealed_indices`
        // as the latter includes adjacent table rows relative to the values in `indices`
        let revealed_combination_elements: Vec<XFieldElement> =
            indices.iter().map(|i| combination_codeword[*i]).collect();
        let revealed_combination_auth_paths = combination_tree.get_multi_proof(&indices);
        proof_stream.enqueue(&Item::RevealedCombinationElements(
            revealed_combination_elements,
        ));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(
            revealed_combination_auth_paths,
        ));

        timer.elapsed("open combination codeword at same positions");

        let report = timer.finish();
        println!("{}", report);
        println!(
            "Created proof containing {} B-field elements",
            proof_stream.transcript_length()
        );

        proof_stream
    }

    pub fn verify(&self, proof_stream: &mut StarkProofStream) -> Result<bool, Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        let hasher = StarkHasher::new();

        let base_merkle_tree_root: Vec<BFieldElement> = proof_stream.dequeue()?.as_merkle_root()?;

        let seed = proof_stream.verifier_fiat_shamir();

        timer.elapsed("verifier_fiat_shamir");
        // Get coefficients for table extension
        let challenge_weights: [XFieldElement; AllChallenges::TOTAL_CHALLENGES] =
            Self::sample_weights(&hasher, &seed, AllChallenges::TOTAL_CHALLENGES)
                .try_into()
                .unwrap();
        let all_challenges: AllChallenges = AllChallenges::create_challenges(&challenge_weights);

        let extension_tree_merkle_root: Vec<BFieldElement> =
            proof_stream.dequeue()?.as_merkle_root()?;

        let all_terminals = proof_stream.dequeue()?.as_terminals()?;
        timer.elapsed("Read from proof stream");

        let max_padded_height = proof_stream
            .dequeue()?
            .as_max_padded_table_height()?
            .value() as usize;
        let ext_table_collection = ExtTableCollection::with_padded_height(
            self.generator,
            self.order,
            self.num_randomizers,
            max_padded_height,
        );

        let base_degree_bounds: Vec<Degree> = ext_table_collection.get_all_base_degree_bounds();
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds: Vec<Degree> =
            ext_table_collection.get_all_extension_degree_bounds();
        timer.elapsed("Calculated extension degree bounds");

        // get weights for nonlinear combination
        //  - 1 randomizer
        //  - 2 for every other polynomial (base, extension, quotients)
        let num_base_polynomials = base_degree_bounds.len();
        let num_extension_polynomials = extension_degree_bounds.len();
        let num_randomizer_polynomials = 1;
        let num_quotient_polynomials: usize = ext_table_collection
            .into_iter()
            .map(|table| {
                table
                    .all_quotient_degree_bounds(&all_challenges, &all_terminals)
                    .len()
            })
            .sum();
        let num_difference_quotients = PermArg::all_permutation_arguments().len();
        timer.elapsed("Calculated quotient degree bounds");

        let weights_seed: Vec<BFieldElement> = proof_stream.verifier_fiat_shamir();

        timer.elapsed("verifier_fiat_shamir (again)");
        let weights_count = num_randomizer_polynomials
            + 2 * num_base_polynomials
            + 2 * num_extension_polynomials
            + 2 * num_quotient_polynomials
            + 2 * num_difference_quotients;
        let weights: Vec<XFieldElement> =
            Self::sample_weights(&hasher, &weights_seed, weights_count);
        timer.elapsed("Calculated weights");

        let combination_root: Vec<BFieldElement> = proof_stream.dequeue()?.as_merkle_root()?;

        let indices_seed: Vec<BFieldElement> = proof_stream.verifier_fiat_shamir();
        let indices =
            hasher.sample_indices(self.security_level, &indices_seed, self.fri.domain.length);
        timer.elapsed("Got indices");

        // Verify low degree of combination polynomial
        self.fri.verify(proof_stream, &combination_root)?;
        timer.elapsed("Verified FRI proof");

        // TODO: Consider factoring out code to find `unit_distances`, duplicated in prover
        let mut unit_distances: Vec<usize> = ext_table_collection
            .into_iter()
            .map(|table| table.unit_distance(self.fri.domain.length))
            .collect();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();
        timer.elapsed("Got unit distances");

        let mut tuples: HashMap<usize, Vec<XFieldElement>> = HashMap::new();
        // TODO: we can store the elements mushed into "tuples" separately, like in "points" below,
        // to avoid unmushing later

        // Open leafs of zipped codewords at indicated positions
        let mut revealed_indices: Vec<usize> = vec![];
        for index in indices.iter() {
            for unit_distance in unit_distances.iter() {
                let idx: usize = (index + unit_distance) % self.fri.domain.length;
                revealed_indices.push(idx);
            }
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();
        timer.elapsed("Calculated revealed indices");

        let revealed_base_elements: Vec<Vec<BFieldElement>> = proof_stream
            .dequeue()?
            .as_transposed_base_element_vectors()?;
        let auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read base elements and auth paths from proof stream");
        let leaf_digests: Vec<Vec<BFieldElement>> = revealed_base_elements
            .par_iter()
            .map(|re| hasher.hash(re, RP_DEFAULT_OUTPUT_SIZE))
            .collect();
        timer.elapsed(&format!(
            "Calculated {} leaf digests for base elements",
            indices.len()
        ));
        let mt_base_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            base_merkle_tree_root,
            &revealed_indices,
            &leaf_digests,
            &auth_paths,
        );
        if !mt_base_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for base codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified authentication paths for {} base elements",
            indices.len()
        ));

        // Get extension elements
        let revealed_extension_elements: Vec<Vec<XFieldElement>> = proof_stream
            .dequeue()?
            .as_transposed_extension_element_vectors()?;
        let extension_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read extension elements and auth paths from proof stream");
        let extension_leaf_digests: Vec<Vec<BFieldElement>> = revealed_extension_elements
            .clone()
            .into_par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .into_iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();
                debug_assert_eq!(
                    27,
                    bvalues.len(),
                    "9 X-field elements must become 27 B-field elements"
                );
                hasher.hash(&bvalues, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        timer.elapsed(&format!(
            "Calculated {} leaf digests for extension elements",
            indices.len()
        ));
        let mt_extension_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            extension_tree_merkle_root,
            &revealed_indices,
            &extension_leaf_digests,
            &extension_auth_paths,
        );
        if !mt_extension_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for extension codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified authentication paths for {} extension elements",
            indices.len()
        ));

        // Collect values in a hash map
        for (i, &idx) in revealed_indices.iter().enumerate() {
            let randomizer: XFieldElement = XFieldElement::new([
                revealed_base_elements[i][0],
                revealed_base_elements[i][1],
                revealed_base_elements[i][2],
            ]);
            debug_assert_eq!(
                1, num_randomizer_polynomials,
                "For now number of randomizers must be 1"
            );
            let mut values: Vec<XFieldElement> = vec![randomizer];
            values.extend_from_slice(
                &revealed_base_elements[i]
                    .iter()
                    .skip(3 * num_randomizer_polynomials)
                    .map(|bfe| bfe.lift())
                    .collect::<Vec<XFieldElement>>(),
            );
            tuples.insert(idx, values);
            tuples.insert(
                idx,
                vec![tuples[&idx].clone(), revealed_extension_elements[i].clone()].concat(),
            );
        }
        timer.elapsed(&format!(
            "Collected {} values into a hash map",
            indices.len()
        ));

        // Verify Merkle authentication path for combination elements
        let revealed_combination_elements: Vec<XFieldElement> =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<Vec<BFieldElement>> = revealed_combination_elements
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let b_elements: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&b_elements, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        let revealed_combination_auth_paths: Vec<PartialAuthenticationPath<Vec<BFieldElement>>> =
            proof_stream
                .dequeue()?
                .as_compressed_authentication_paths()?;
        let mt_combination_success = MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            combination_root.clone(),
            &indices,
            &revealed_combination_digests,
            &revealed_combination_auth_paths,
        );
        if !mt_combination_success {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for combination codeword");
            // return Ok(false);
        }
        timer.elapsed(&format!(
            "Verified combination authentication paths for {} indices",
            indices.len()
        ));

        // verify nonlinear combination
        for (i, &index) in indices.iter().enumerate() {
            let b_domain_value = self.fri.domain.b_domain_value(index as u32);
            // collect terms: randomizer
            let mut terms: Vec<XFieldElement> = (0..num_randomizer_polynomials)
                .map(|j| tuples[&index][j])
                .collect();

            // collect terms: base
            for j in num_randomizer_polynomials..num_randomizer_polynomials + num_base_polynomials {
                terms.push(tuples[&index][j]);
                let shift: u32 = (self.max_degree as i64
                    - base_degree_bounds[j - num_randomizer_polynomials])
                    as u32;
                terms.push(tuples[&index][j] * b_domain_value.mod_pow_u32(shift).lift());
            }

            // collect terms: extension
            let extension_offset = num_randomizer_polynomials + num_base_polynomials;

            assert_eq!(
                terms.len(),
                2 * extension_offset - num_randomizer_polynomials,
                "number of terms does not match with extension offset"
            );

            // TODO: We don't seem to need a separate loop for the base and extension columns.
            // But merging them would also require concatenating the degree bounds vector.
            for (j, edb) in extension_degree_bounds.iter().enumerate() {
                let extension_element: XFieldElement = tuples[&index][extension_offset + j];
                terms.push(extension_element);
                let shift = (self.max_degree as i64 - edb) as u32;
                terms.push(extension_element * b_domain_value.mod_pow_u32(shift).lift())
            }

            // collect terms: quotients, quotients need to be computed
            let mut acc_index = num_randomizer_polynomials;
            let mut points: Vec<Vec<XFieldElement>> = vec![];
            for table in ext_table_collection.into_iter() {
                let table_base_width = table.base_width();
                points.push(tuples[&index][acc_index..acc_index + table_base_width].to_vec());
                acc_index += table_base_width;
            }

            assert_eq!(
                extension_offset, acc_index,
                "Column count in verifier must match until extension columns"
            );

            for (point, table) in points.iter_mut().zip(ext_table_collection.into_iter()) {
                let step_size = table.width() - table.base_width();
                point.extend_from_slice(&tuples[&index][acc_index..acc_index + step_size]);
                acc_index += step_size;
            }

            assert_eq!(
                tuples[&index].len(),
                acc_index,
                "Column count in verifier must match until end"
            );

            let mut base_acc_index = num_randomizer_polynomials;
            let mut ext_acc_index = extension_offset;
            for (point, table) in points.iter().zip(ext_table_collection.into_iter()) {
                // boundary
                for (constraint, bound) in
                    table.ext_boundary_constraints(&all_challenges).iter().zip(
                        table
                            .boundary_quotient_degree_bounds(&all_challenges)
                            .iter(),
                    )
                {
                    let eval = constraint.evaluate(point);
                    let quotient = eval / (b_domain_value.lift() - XFieldElement::ring_one());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
                }

                // transition
                let unit_distance = table.unit_distance(self.fri.domain.length);
                let next_index = (index + unit_distance) % self.fri.domain.length;
                let mut next_point = tuples[&next_index]
                    [base_acc_index..base_acc_index + table.base_width()]
                    .to_vec();
                next_point.extend_from_slice(
                    &tuples[&next_index]
                        [ext_acc_index..ext_acc_index + table.width() - table.base_width()],
                );
                base_acc_index += table.base_width();
                ext_acc_index += table.width() - table.base_width();
                for (constraint, bound) in table
                    .ext_transition_constraints(&all_challenges)
                    .iter()
                    .zip(
                        table
                            .transition_quotient_degree_bounds(&all_challenges)
                            .iter(),
                    )
                {
                    let eval =
                        constraint.evaluate(&vec![point.to_owned(), next_point.clone()].concat());
                    // If height == 0, then there is no subgroup where the transition polynomials should be zero.
                    // The fast zerofier (based on group theory) needs a non-empty group.
                    // Forcing it on an empty group generates a division by zero error.
                    let quotient = if table.padded_height() == 0 {
                        XFieldElement::ring_zero()
                    } else {
                        // TODO: This is probably wrong
                        let num = b_domain_value.lift() - table.omicron().inverse();
                        let denom = b_domain_value
                            .mod_pow_u32(table.padded_height() as u32)
                            .lift()
                            - XFieldElement::ring_one();
                        eval * num / denom
                    };
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
                }

                // terminal
                for (constraint, bound) in table
                    .ext_terminal_constraints(&all_challenges, &all_terminals)
                    .iter()
                    .zip(
                        table
                            .terminal_quotient_degree_bounds(&all_challenges, &all_terminals)
                            .iter(),
                    )
                {
                    let eval = constraint.evaluate(point);
                    // TODO: Removed lift()
                    let quotient = eval / (b_domain_value.lift() - table.omicron().inverse());
                    terms.push(quotient);
                    let shift = (self.max_degree as i64 - bound) as u32;
                    terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift())
                }
            }

            for arg in PermArg::all_permutation_arguments().iter() {
                let quotient = arg.evaluate_difference(&points)
                    / (b_domain_value.lift() - XFieldElement::ring_one());
                terms.push(quotient);
                let degree_bound = arg.quotient_degree_bound(&ext_table_collection);
                let shift = (self.max_degree as i64 - degree_bound) as u32;
                terms.push(quotient * b_domain_value.mod_pow_u32(shift).lift());
            }

            assert_eq!(
                weights.len(),
                terms.len(),
                "length of terms must be equal to length of weights"
            );

            // compute inner product of weights and terms
            // Todo: implement `sum` on XFieldElements
            let inner_product = weights
                .par_iter()
                .zip(terms.par_iter())
                .map(|(w, t)| *w * *t)
                .reduce(XFieldElement::ring_zero, |x, y| x + y);

            assert_eq!(
                revealed_combination_elements[i], inner_product,
                "The combination leaf must equal the inner product"
            );
        }
        timer.elapsed(&format!(
            "Verified {} non-linear combinations",
            indices.len()
        ));

        // Verify external terminals
        if !all_evaluation_arguments(&[], &[], &all_challenges, &all_terminals) {
            return Err(Box::new(StarkVerifyError::EvaluationArgument(0)));
        }

        timer.elapsed("Verified terminals");

        let report = timer.finish();
        println!("{}", report);

        Ok(true)
    }

    // FIXME: This interface leaks abstractions: We want a function that generates a number of weights
    // that doesn't care about the weights-to-digest ratio (we can make two weights per digest).
    fn sample_weights(
        hasher: &StarkHasher,
        seed: &StarkDigest,
        count: usize,
    ) -> Vec<XFieldElement> {
        hasher
            .get_n_hash_rounds(seed, count / 2)
            .iter()
            .flat_map(|digest| {
                vec![
                    XFieldElement::new([digest[0], digest[1], digest[2]]),
                    XFieldElement::new([digest[3], digest[4], digest[5]]),
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod triton_stark_tests {
    use crate::shared_math::stark::triton::instruction::parse;
    use crate::shared_math::stark::triton::stdio::VecStream;

    use super::*;

    fn parse_simulate_pad_extend(
        code: &str,
    ) -> (BaseTableCollection, BaseTableCollection, ExtTableCollection, AllEndpoints) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let mut _rng = rand::thread_rng();
        let mut stdin = VecStream::new(&[]);
        let mut secret_in = VecStream::new(&[]);
        let mut stdout = VecStream::new(&[]);
        let rescue_prime = neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);

        assert!(err.is_none(), "simulate did not generate errors");

        let num_randomizers = 2;
        let order: usize = 1 << 32;
        let smooth_generator = BFieldElement::ring_zero()
            .get_primitive_root_of_unity(order as u64)
            .0
            .unwrap();

        let mut base_tables = BaseTableCollection::from_base_matrices(
            smooth_generator,
            order,
            num_randomizers,
            &base_matrices,
        );

        let unpadded_base_tables = base_tables.clone();

        base_tables.pad();

        let hasher = StarkHasher::new();
        let mock_seed = hasher.hash(&[], DIGEST_LEN);

        let challenge_weights =
            Stark::sample_weights(&hasher, &mock_seed, AllChallenges::TOTAL_CHALLENGES);
        let all_challenges: AllChallenges = AllChallenges::create_challenges(&challenge_weights);

        let initial_weights =
            Stark::sample_weights(&hasher, &mock_seed, AllEndpoints::TOTAL_ENDPOINTS);
        let all_initials: AllEndpoints = AllEndpoints::create_initials(&initial_weights);

        let (ext_tables, all_terminals) =
            ExtTableCollection::extend_tables(&base_tables, &all_challenges, &all_initials);

        (base_tables, ext_tables, all_terminals)
    }

    // 1. simulate(), pad(), extend(), test terminals
    // 2. simulate(), test constraints
    // 3. simulate(), pad(), test constraints
    // 3. simulate(), pad(), extend(), test constraints
}
