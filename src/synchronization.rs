use smallvec::SmallVec;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CommandBufferSubmission(u32);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReaderWriterState {
    Read(SmallVec<[CommandBufferSubmission; 4]>),
    Write(CommandBufferSubmission),
    None,
}
