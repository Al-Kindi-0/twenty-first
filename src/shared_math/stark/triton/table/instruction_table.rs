use super::base_table::{self, BaseTable, HasBaseTable, Table};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::table_column::InstructionTableColumn::{self, *};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::MPolynomial;
use crate::shared_math::other;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::x_field_element::XFieldElement;

pub const INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 1;
pub const INSTRUCTION_TABLE_INITIALS_COUNT: usize =
    INSTRUCTION_TABLE_PERMUTATION_ARGUMENTS_COUNT + INSTRUCTION_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 6 because it combines: (ip, ci, nia) and (addr, instruction, next_instruction).
pub const INSTRUCTION_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 6;

pub const BASE_WIDTH: usize = 3;
pub const FULL_WIDTH: usize = 7; // BASE_WIDTH + 2 * INITIALS_COUNT

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone)]
pub struct InstructionTable {
    base: BaseTable<BWord>,
}

impl HasBaseTable<BWord> for InstructionTable {
    fn to_base(&self) -> &BaseTable<BWord> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<BWord> {
        &mut self.base
    }
}

#[derive(Debug, Clone)]
pub struct ExtInstructionTable {
    base: BaseTable<XFieldElement>,
}

impl HasBaseTable<XFieldElement> for ExtInstructionTable {
    fn to_base(&self) -> &BaseTable<XFieldElement> {
        &self.base
    }

    fn to_mut_base(&mut self) -> &mut BaseTable<XFieldElement> {
        &mut self.base
    }
}

impl Table<BWord> for InstructionTable {
    fn name(&self) -> String {
        "InstructionTable".to_string()
    }

    fn pad(&mut self) {
        let data = self.mut_data();
        while !data.is_empty() && !other::is_power_of_two(data.len()) {
            let mut padding_row = data.last().unwrap().clone();
            // address keeps increasing
            padding_row[InstructionTableColumn::Address as usize] += 1.into();
            data.push(padding_row);
        }
    }
}

impl Table<XFieldElement> for ExtInstructionTable {
    fn name(&self) -> String {
        "ExtInstructionTable".to_string()
    }

    fn pad(&mut self) {
        panic!("Extension tables don't get padded");
    }
}

impl ExtensionTable for ExtInstructionTable {
    fn ext_boundary_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(FULL_WIDTH, 1.into());
        let addr = variables[usize::from(Address)].clone();

        // The first address is 0.
        let fst_addr_is_zero = addr;

        vec![fst_addr_is_zero]
    }

    fn ext_consistency_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(&self, _challenges: &AllChallenges) -> Vec<MPolynomial<XWord>> {
        let variables: Vec<MPolynomial<XWord>> = MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        let addr = variables[usize::from(Address)].clone();
        let addr_next = variables[FULL_WIDTH + usize::from(Address)].clone();
        let current_instruction = variables[CI as usize].clone();
        let current_instruction_next = variables[FULL_WIDTH + CI as usize].clone();
        let next_instruction = variables[NIA as usize].clone();
        let next_instruction_next = variables[FULL_WIDTH + NIA as usize].clone();
        let one = MPolynomial::<XFieldElement>::from_constant(1.into(), 2 * FULL_WIDTH);

        let addr_incr_by_one = addr_next - (addr + one);

        // The address increases by 1 or `current_instruction` does not change.
        let addr_incr_by_one_or_ci_stays =
            addr_incr_by_one.clone() * (current_instruction_next - current_instruction);

        // The address increases by 1 or `next_instruction_or_arg` does not change.
        let addr_incr_by_one_or_nia_stays =
            addr_incr_by_one * (next_instruction_next - next_instruction);

        vec![addr_incr_by_one_or_ci_stays, addr_incr_by_one_or_nia_stays]
    }

    fn ext_terminal_constraints(
        &self,
        _challenges: &AllChallenges,
        _terminals: &AllEndpoints,
    ) -> Vec<MPolynomial<XWord>> {
        // no further constraints
        vec![]
    }
}

impl InstructionTable {
    pub fn new_prover(
        generator: BWord,
        order: usize,
        num_randomizers: usize,
        matrix: Vec<Vec<BWord>>,
    ) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::pad_height(unpadded_height);

        let dummy = generator;
        let omicron = base_table::derive_omicron(padded_height as u64, dummy);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        );

        Self { base }
    }

    pub fn extend(
        &self,
        challenges: &InstructionTableChallenges,
        initials: &InstructionTableEndpoints,
    ) -> (ExtInstructionTable, InstructionTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut processor_table_running_product = initials.processor_perm_product;
        let mut program_table_running_sum = initials.program_eval_sum;
        let mut previous_row: Option<Vec<_>> = None;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Is the current row's address different from the previous row's address?
            // Different: update running sum of Evaluation Argument with Program Table.
            // Not different: update running product of Permutation Argument with Processor Table.
            let mut is_duplicate_row = false;
            if let Some(prow) = previous_row {
                if prow[Address as usize] == row[Address as usize] {
                    is_duplicate_row = true;
                    debug_assert_eq!(prow[CI as usize], row[CI as usize]);
                    debug_assert_eq!(prow[NIA as usize], row[NIA as usize]);
                }
            }

            // Compress values of current row for Permutation Argument with Processor Table
            let ip = row[Address as usize].lift();
            let ci = row[CI as usize].lift();
            let nia = row[NIA as usize].lift();
            let compressed_row_for_permutation_argument = ip * challenges.ip_processor_weight
                + ci * challenges.ci_processor_weight
                + nia * challenges.nia_processor_weight;
            extension_row.push(compressed_row_for_permutation_argument);

            // Update running product for Permutation Argument if same row has been seen before
            extension_row.push(processor_table_running_product);
            if is_duplicate_row {
                processor_table_running_product *=
                    challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;
            }

            // Compress values of current row for Evaluation Argument with Program Table
            let address = row[Address as usize].lift();
            let instruction = row[CI as usize].lift();
            let next_instruction = row[NIA as usize].lift();
            let compressed_row_for_evaluation_argument = address * challenges.address_weight
                + instruction * challenges.instruction_weight
                + next_instruction * challenges.next_instruction_weight;
            extension_row.push(compressed_row_for_evaluation_argument);

            // Update running sum for Evaluation Argument if same row has _not_ been seen before
            extension_row.push(program_table_running_sum);
            if !is_duplicate_row {
                program_table_running_sum = program_table_running_sum
                    * challenges.program_eval_row_weight
                    + compressed_row_for_evaluation_argument;
            }

            debug_assert_eq!(
                FULL_WIDTH,
                extension_row.len(),
                "After extending, the row must match the table's full width."
            );

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row);
        }

        let base = self.base.with_lifted_data(extension_matrix);
        let table = ExtInstructionTable { base };
        let terminals = InstructionTableEndpoints {
            processor_perm_product: processor_table_running_product,
            program_eval_sum: program_table_running_sum,
        };

        (table, terminals)
    }
}

impl ExtInstructionTable {
    pub fn with_padded_height(
        generator: XWord,
        order: usize,
        num_randomizers: usize,
        padded_height: usize,
    ) -> Self {
        let matrix: Vec<Vec<XWord>> = vec![];

        let dummy = generator;
        let omicron = base_table::derive_omicron(padded_height as u64, dummy);
        let base = BaseTable::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_randomizers,
            omicron,
            generator,
            order,
            matrix,
        );

        Self { base }
    }

    pub fn ext_codeword_table(&self, fri_domain: &FriDomain<XWord>) -> Self {
        let ext_codewords = self.low_degree_extension(fri_domain, self.full_width());
        let base = self.base.with_data(ext_codewords);

        ExtInstructionTable { base }
    }
}

#[derive(Debug, Clone)]
pub struct InstructionTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub ip_processor_weight: XFieldElement,
    pub ci_processor_weight: XFieldElement,
    pub nia_processor_weight: XFieldElement,

    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the instruction table.
    pub program_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to program table.)
    pub address_weight: XFieldElement,
    pub instruction_weight: XFieldElement,
    pub next_instruction_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct InstructionTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
    pub program_eval_sum: XFieldElement,
}
