use std::error::Error;
use std::fmt::{Display, Formatter};
use InstructionError::*;

#[derive(Debug, Clone)]
pub enum InstructionError {
    InstructionPointerOverflow(usize),
    OpStackTooShallow,
    JumpStackTooShallow,
    AssertionFailed,
    MemoryAddressNotFound,
    InverseOfZero,
    RunawayInstructionArg,
    UngracefulTermination,
}

impl Display for InstructionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionPointerOverflow(ip) => {
                write!(f, "Instruction pointer {} points outside of program", ip)
            }

            OpStackTooShallow => {
                write!(f, "Instruction addresses too deeply into the stack")
            }

            JumpStackTooShallow => {
                write!(f, "Jump stack does not contain return address")
            }

            AssertionFailed => {
                write!(f, "Assertion failed")
            }

            MemoryAddressNotFound => {
                write!(f, "Memory address not found")
            }

            InverseOfZero => {
                write!(f, "0 does not have a multiplicative inverse")
            }

            RunawayInstructionArg => {
                write!(
                    f,
                    "A numeric argument to an instruction occurred out of place"
                )
            }

            UngracefulTermination => {
                write!(
                    f,
                    "The Virtual Machine must terminate using instruction Halt"
                )
            }
        }
    }
}

impl Error for InstructionError {}

pub fn vm_err<T>(runtime_error: InstructionError) -> Result<T, Box<dyn Error>> {
    Err(vm_fail(runtime_error))
}

pub fn vm_fail(runtime_error: InstructionError) -> Box<dyn Error> {
    Box::new(runtime_error)
}
