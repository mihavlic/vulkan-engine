use std::sync::Arc;

use pumice::{
    loader::{tables::InstanceTable, EntryLoader},
    vk10::Instance,
};

pub(crate) struct InnerInstance {
    pub(crate) entry: pumice::EntryWrapper,
    pub(crate) instance: pumice::InstanceWrapper,

    pub(crate) entry_table: EntryTable,
    pub(crate) instance_table: InstanceTable,
    // at the bottom so that it is dropped last, the loader keeps the vulkan dll loaded
    pub(crate) loader: EntryLoader,
}

#[derive(Clone)]
pub struct Instance(Arc<InnerInstance>);
