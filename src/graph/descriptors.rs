use std::{
    num::NonZeroUsize,
    ops::{Deref, DerefMut, Range},
    ptr::NonNull,
};

use bumpalo::Bump;
use pumice::{util::ObjectHandle, vk, DeviceWrapper, VulkanResult};
use smallvec::SmallVec;

use crate::{
    device::{
        debug::maybe_attach_debug_label,
        ring::{BufferEntry, QueueRing, RingConfig, SuballocatedMemory},
        Device,
    },
    graph::allocator::round_up_pow2_usize,
    object::{self, DescriptorBinding, ObjRef},
};

use super::execute::GraphExecutor;

#[macro_export]
macro_rules! desc_set_sizes {
    ($($count:literal * $kind:ident),+ $(,)?) => {
        &[
            $(
                vk::DescriptorPoolSize {
                    kind: vk::DescriptorType::$kind,
                    descriptor_count: $count,
                }
            ),+
        ]
    };
}

// the minimum required value of maxUniformBufferRange
const UNIFORM_BUFFER_SIZE: u64 = 16384;

const DESCRIPTOR_SET_SIZES: &[vk::DescriptorPoolSize] = desc_set_sizes!(
    256 * SAMPLER,
    256 * COMBINED_IMAGE_SAMPLER,
    256 * SAMPLED_IMAGE,
    256 * STORAGE_IMAGE,
    256 * UNIFORM_TEXEL_BUFFER,
    256 * STORAGE_TEXEL_BUFFER,
    256 * UNIFORM_BUFFER,
    256 * STORAGE_BUFFER,
    256 * UNIFORM_BUFFER_DYNAMIC,
    256 * STORAGE_BUFFER_DYNAMIC,
    256 * INPUT_ATTACHMENT,
);

pub struct DescriptorSetAllocator {
    free_pools: Vec<vk::DescriptorPool>,
    pools: Vec<vk::DescriptorPool>,
    sizes: &'static [vk::DescriptorPoolSize],
}

impl DescriptorSetAllocator {
    pub fn new(sizes: &'static [vk::DescriptorPoolSize]) -> Self {
        Self {
            free_pools: Vec::new(),
            pools: Vec::new(),
            sizes,
        }
    }
    pub unsafe fn allocate_set(
        &mut self,
        layout: &ObjRef<object::DescriptorSetLayout>,
        device: &Device,
    ) -> vk::DescriptorSet {
        if self.pools.is_empty() {
            self.add_descriptor_pool(device);
        }

        let pool = *self.pools.last().unwrap();
        let layout = layout.get_handle();

        let info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            descriptor_set_count: 1,
            p_set_layouts: &layout,
            ..Default::default()
        };

        let result = device.device().allocate_descriptor_sets(&info);
        let set = result.unwrap_or_else(|err| {
            match err {
                vk::Result::ERROR_OUT_OF_HOST_MEMORY
                | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                | vk::Result::ERROR_FRAGMENTED_POOL
                | vk::Result::ERROR_OUT_OF_POOL_MEMORY => {}
                _ => panic!("Unexpected error from allocate_descriptor_sets: {:?}", err),
            }

            self.add_descriptor_pool(device);
            device
                .device()
                .allocate_descriptor_sets(&info)
                .expect("Unable to allocate set in a fresh pool")
        });

        set[0]
    }
    unsafe fn add_descriptor_pool(&mut self, device: &Device) {
        if let Some(free) = self.free_pools.pop() {
            self.pools.push(free);
            return;
        }

        let info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::empty(),
            max_sets: 512,
            pool_size_count: self.sizes.len() as u32,
            p_pool_sizes: self.sizes.as_ptr(),
            ..Default::default()
        };
        let pool = device
            .device()
            .create_descriptor_pool(&info, device.allocator_callbacks())
            .unwrap();
        self.pools.push(pool);
    }
    pub unsafe fn reset(&mut self, device: &Device) {
        self.free_pools.extend(self.pools.drain(..).map(|p| {
            device.device().reset_descriptor_pool(p, None);
            p
        }));
    }
    pub unsafe fn destroy(&mut self, device: &Device) {
        for &pool in self.pools.iter().chain(&self.free_pools) {
            device
                .device()
                .destroy_descriptor_pool(pool, device.allocator_callbacks());
        }
    }
}

pub struct UniformAllocator {
    buffers: QueueRing,
}

pub struct UniformSetAllocator {
    pub sets: DescriptorSetAllocator,
    pub uniforms: UniformAllocator,
}

impl UniformSetAllocator {
    pub fn new() -> Self {
        Self {
            sets: DescriptorSetAllocator::new(DESCRIPTOR_SET_SIZES),
            uniforms: UniformAllocator::new(),
        }
    }
    pub(crate) unsafe fn reset(&mut self, device: &Device) {
        self.sets.reset(device);
        self.uniforms.reset();
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        self.sets.destroy(device);
        self.uniforms.destroy(device);
    }
}

impl UniformAllocator {
    pub(crate) fn new() -> Self {
        const CONFIG: RingConfig = RingConfig {
            buffer_size: 16384,
            usage: vk::BufferUsageFlags(
                vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER.0
                    | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER.0
                    | vk::BufferUsageFlags::UNIFORM_BUFFER.0
                    | vk::BufferUsageFlags::STORAGE_BUFFER.0,
            ),
            allocation_flags: pumice_vma::AllocationCreateFlags(
                pumice_vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE.0
                    | pumice_vma::AllocationCreateFlags::MAPPED.0,
            ),
            required_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
            preferred_flags: vk::MemoryPropertyFlags::empty(),
            label: "DescriptorAllocator buffer",
        };

        Self {
            buffers: QueueRing::new(&CONFIG),
        }
    }
    pub(crate) unsafe fn reset(&mut self) {
        self.buffers.reset_all();
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        self.buffers.destroy(device);
    }
    pub unsafe fn allocate_uniform_iter<T, I: IntoIterator<Item = T>>(
        &mut self,
        iter: I,
        device: &Device,
    ) -> UniformResult
    where
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        assert!(iter.len() > 0);

        let layout = std::alloc::Layout::array::<T>(iter.len()).unwrap();
        let uniform = self.allocate_uniform_raw(layout, device);

        let SuballocatedMemory {
            dynamic_offset,
            buffer,
            memory,
        } = uniform;
        let ptr = memory.as_ptr().cast::<T>();

        for (i, item) in iter.enumerate() {
            *ptr.add(i) = item;
        }

        UniformResult {
            dynamic_offset,
            buffer,
        }
    }
    pub(crate) unsafe fn allocate_uniform_element<T>(
        &mut self,
        value: &T,
        device: &Device,
    ) -> UniformResult {
        let layout = std::alloc::Layout::new::<T>();
        let uniform = self.allocate_uniform_raw(layout, device);

        let SuballocatedMemory {
            dynamic_offset,
            buffer,
            memory,
        } = uniform;
        let ptr = memory.as_ptr().cast::<T>();
        std::ptr::copy_nonoverlapping(value, ptr, 1);

        UniformResult {
            dynamic_offset,
            buffer,
        }
    }
    pub(crate) unsafe fn allocate_uniform_raw(
        &mut self,
        layout: std::alloc::Layout,
        device: &Device,
    ) -> SuballocatedMemory {
        self.buffers.allocate(layout, device)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct UniformResult {
    pub dynamic_offset: u32,
    pub buffer: vk::Buffer,
}

#[derive(Default, Clone, PartialEq, Eq)]
pub struct DescImage {
    pub sampler: vk::Sampler,
    pub view: vk::ImageView,
    pub layout: vk::ImageLayout,
}

#[derive(Clone)]
pub struct DescBuffer {
    pub buffer: vk::Buffer,
    pub view: vk::BufferView,
    pub offset: u64,
    pub range: u64,
    pub dynamic_offset: Option<u32>,
}

impl Default for DescBuffer {
    fn default() -> Self {
        Self {
            buffer: vk::Buffer::null(),
            view: vk::BufferView::null(),
            offset: 0,
            range: vk::WHOLE_SIZE,
            dynamic_offset: None,
        }
    }
}

impl DescBuffer {
    /// Compares all fields of DescBuffer except for dynamic_offset
    fn set_equal(&self, other: &Self) -> bool {
        self.buffer == other.buffer
            && self.view == other.view
            && self.offset == other.offset
            && self.range == other.range
    }
}

pub enum DescriptorData<'a> {
    Image(DescImage),
    Buffer(DescBuffer),
    ImageArr(&'a [DescImage]),
    BufferArr(&'a [DescBuffer]),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DescriptorCategory {
    Image,
    Buffer,
}

impl DescriptorCategory {
    fn from_kind(kind: vk::DescriptorType) -> Self {
        match kind {
            vk::DescriptorType::SAMPLER => Self::Image,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER => Self::Image,
            vk::DescriptorType::SAMPLED_IMAGE => Self::Image,
            vk::DescriptorType::STORAGE_IMAGE => Self::Image,
            vk::DescriptorType::UNIFORM_TEXEL_BUFFER => Self::Buffer,
            vk::DescriptorType::STORAGE_TEXEL_BUFFER => Self::Buffer,
            vk::DescriptorType::UNIFORM_BUFFER => Self::Buffer,
            vk::DescriptorType::STORAGE_BUFFER => Self::Buffer,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => Self::Buffer,
            vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => Self::Buffer,
            vk::DescriptorType::INPUT_ATTACHMENT => Self::Image,
            _ => unimplemented!(),
        }
    }
}

impl From<vk::DescriptorType> for DescriptorCategory {
    fn from(value: vk::DescriptorType) -> Self {
        Self::from_kind(value)
    }
}

#[derive(Clone)]
struct SetBindingState {
    data_slice: Range<u32>,
    category: DescriptorCategory,
}

#[derive(Clone)]
pub struct DescSetBuilder<'a> {
    set: vk::DescriptorSet,
    layout: &'a ObjRef<object::DescriptorSetLayout>,

    bindings: Box<[SetBindingState]>,
    images: Box<[DescImage]>,
    buffers: Box<[DescBuffer]>,
}

impl<'a> DescSetBuilder<'a> {
    pub fn new(layout: &'a ObjRef<object::DescriptorSetLayout>) -> Self {
        let bindings = &layout.get_create_info().bindings;

        let mut image_count = 0;
        let mut buffer_count = 0;
        let bindings = bindings
            .iter()
            .map(|b| {
                let category: DescriptorCategory = b.kind.into();
                let interval = match category {
                    DescriptorCategory::Image => {
                        let start = image_count;
                        image_count += b.count;
                        start..image_count
                    }
                    DescriptorCategory::Buffer => {
                        let start = buffer_count;
                        buffer_count += b.count;
                        start..buffer_count
                    }
                };
                SetBindingState {
                    data_slice: interval,
                    category,
                }
            })
            .collect();

        let mut images = vec![Default::default(); image_count as usize].into_boxed_slice();
        let mut buffers = vec![Default::default(); buffer_count as usize].into_boxed_slice();

        Self {
            set: vk::DescriptorSet::null(),
            layout,
            bindings,
            images,
            buffers,
        }
    }
    fn mark_dirty(&mut self) {
        self.set = vk::DescriptorSet::null();
    }
    pub fn update_image_binding(
        &mut self,
        binding: u32,
        array_offset: u32,
        data: &DescImage,
    ) -> &mut Self {
        self.update_image_bindings_arr(binding, array_offset, std::slice::from_ref(data));
        self
    }
    pub fn update_buffer_binding(
        &mut self,
        binding: u32,
        array_offset: u32,
        data: &DescBuffer,
    ) -> &mut Self {
        self.update_buffer_bindings_arr(binding, array_offset, std::slice::from_ref(data));
        self
    }
    pub fn update_buffer_binding_dynamic(
        &mut self,
        binding: u32,
        array_offset: u32,
        result: UniformResult,
    ) -> &mut Self {
        self.update_buffer_binding(
            binding,
            array_offset,
            &DescBuffer {
                buffer: result.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
                dynamic_offset: Some(result.dynamic_offset),
                ..Default::default()
            },
        );
        self
    }
    pub fn update_image_bindings_arr(
        &mut self,
        binding: u32,
        array_offset: u32,
        datas: &[DescImage],
    ) -> &mut Self {
        let binding = &self.bindings[binding as usize];
        assert!(binding.category == DescriptorCategory::Image);
        let data_slice = binding.data_slice.start as usize..binding.data_slice.end as usize;

        for (i, data) in datas.iter().enumerate() {
            let mut dirty = false;
            let prev_data = &mut self.images[data_slice.clone()][array_offset as usize + i];
            if prev_data != data {
                dirty = true;
                *prev_data = data.clone();
            }
            if dirty {
                self.mark_dirty();
            }
        }
        self
    }
    pub fn update_buffer_bindings_arr(
        &mut self,
        binding: u32,
        array_offset: u32,
        datas: &[DescBuffer],
    ) -> &mut Self {
        let binding = &self.bindings[binding as usize];
        assert!(binding.category == DescriptorCategory::Buffer);
        assert!(array_offset as usize + datas.len() <= binding.data_slice.len());
        let data_slice = binding.data_slice.start as usize..binding.data_slice.end as usize;

        for (i, data) in datas.iter().enumerate() {
            let prev_data = &mut self.buffers[data_slice.clone()][array_offset as usize + i];

            if !prev_data.set_equal(data) {
                *prev_data = data.clone();
                self.mark_dirty();
            }
        }
        self
    }
    pub fn update_whole(&mut self, bindings: &[DescriptorData]) -> &mut Self {
        let layout = &self.layout.get_create_info();
        assert_eq!(bindings.len(), layout.bindings.len());

        let bump = Bump::new();
        for ((desc, data), i) in layout.bindings.iter().zip(bindings).zip(0u32..) {
            let count = desc.count as usize;
            match data {
                DescriptorData::Image(image) => {
                    assert_eq!(count, 1);
                    self.update_image_bindings_arr(i, 0, std::slice::from_ref(image));
                }
                DescriptorData::ImageArr(images) => {
                    assert_eq!(count, images.len());
                    self.update_image_bindings_arr(i, 0, *images);
                }
                DescriptorData::Buffer(buffer) => {
                    assert_eq!(count, 1);
                    self.update_buffer_bindings_arr(i, 0, std::slice::from_ref(buffer));
                }
                DescriptorData::BufferArr(buffers) => {
                    assert_eq!(count, buffers.len());
                    self.update_buffer_bindings_arr(i, 0, *buffers);
                }
            }
        }
        self
    }
    pub unsafe fn finish(&mut self, executor: &GraphExecutor) -> FinishedSet {
        let state = executor.graph.state();
        let mut descriptor_allocator = state.descriptor_allocator.borrow_mut();
        self.finish_internal(&state.device, &mut descriptor_allocator)
    }
    unsafe fn finish_internal(
        &mut self,
        device: &Device,
        allocator: &mut UniformSetAllocator,
    ) -> FinishedSet {
        let mut dynamic_offsets = SmallVec::new();
        for (desc, data) in self
            .layout
            .get_create_info()
            .bindings
            .iter()
            .zip(&*self.bindings)
        {
            if desc.kind == vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                || desc.kind == vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
            {
                let data_slice = data.data_slice.start as usize..data.data_slice.end as usize;
                let data = &self.buffers[data_slice];

                for buffer in data {
                    dynamic_offsets.push(buffer.dynamic_offset.unwrap());
                }
            }
        }

        if self.set == vk::DescriptorSet::null() {
            self.set = allocator.sets.allocate_set(self.layout, device);
            let bindings = &self.layout.get_create_info().bindings;

            write_descriptors(device, &[self]);
        }

        FinishedSet {
            layout: self.layout,
            set: self.set,
            dynamic_offsets,
        }
    }
}

enum DescriptorWriteData<'a> {
    Image(&'a [DescImage]),
    Buffer(&'a [DescBuffer]),
}

unsafe fn write_descriptors(device: &Device, sets: &[&DescSetBuilder]) {
    let bump = Bump::new();

    let mut buffer_views: SmallVec<[vk::BufferView; 32]> = SmallVec::new();

    let writes_count = {
        let mut count = 0;
        for &set in sets {
            for b in set.bindings.deref() {
                count += b.data_slice.end - b.data_slice.start;
            }
        }
        count
    };
    let mut writes_out_offset = 0;
    let writes_out =
        bump.alloc_slice_fill_clone(writes_count as usize, &vk::WriteDescriptorSet::default());

    for &set in sets {
        let bindings = &set.bindings;
        let desc_bindings = &set.layout.get_create_info().bindings;

        for ((write, b), i) in desc_bindings.iter().zip(bindings.deref()).zip(0u32..) {
            let mut p_image_info = std::ptr::null();
            let mut p_buffer_info = std::ptr::null();
            let mut p_texel_buffer_view = std::ptr::null();

            match b.category {
                DescriptorCategory::Image => {
                    let data_slice = b.data_slice.start as usize..b.data_slice.end as usize;
                    let slice = &set.images[data_slice];
                    assert_eq!(slice.len(), write.count as usize);

                    let image_infos = bump.alloc_slice_fill_iter(slice.iter().map(|img| {
                        vk::DescriptorImageInfo {
                            sampler: img.sampler,
                            image_view: img.view,
                            image_layout: img.layout,
                        }
                    }));
                    p_image_info = image_infos.as_ptr();
                }
                DescriptorCategory::Buffer => {
                    let data_slice = b.data_slice.start as usize..b.data_slice.end as usize;
                    let slice = &set.buffers[data_slice];
                    assert_eq!(slice.len(), write.count as usize);

                    buffer_views.clear();
                    let image_infos = bump.alloc_slice_fill_iter(slice.iter().map(|img| {
                        if img.view != vk::BufferView::null() {
                            buffer_views.push(img.view);
                        }
                        vk::DescriptorBufferInfo {
                            buffer: img.buffer,
                            offset: img.offset,
                            range: img.range,
                        }
                    }));
                    let buffer_views = bump.alloc_slice_copy(&buffer_views);

                    p_buffer_info = image_infos.as_ptr();
                    p_texel_buffer_view = buffer_views.as_ptr();
                }
            }

            writes_out[writes_out_offset] = vk::WriteDescriptorSet {
                dst_set: set.set,
                dst_binding: i,
                dst_array_element: 0,
                descriptor_count: write.count,
                descriptor_type: write.kind,
                p_image_info,
                p_buffer_info,
                p_texel_buffer_view,
                ..Default::default()
            };
            writes_out_offset += 1;
        }
    }

    device.device().update_descriptor_sets(&writes_out, &[]);
}

pub struct FinishedSet<'a> {
    layout: &'a ObjRef<object::DescriptorSetLayout>,
    set: vk::DescriptorSet,
    dynamic_offsets: SmallVec<[u32; 4]>,
}

impl<'a> FinishedSet<'a> {
    pub unsafe fn into_raw(self) -> (vk::DescriptorSet, SmallVec<[u32; 4]>) {
        (self.set, self.dynamic_offsets)
    }
    pub unsafe fn bind(
        &self,
        bind_point: vk::PipelineBindPoint,
        layout: &ObjRef<object::PipelineLayout>,
        executor: &GraphExecutor,
    ) {
        executor.bind_descriptor_sets(bind_point, layout, &[self]);
    }
}

pub unsafe fn bind_descriptor_sets(
    device: &DeviceWrapper,
    cmd: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    layout: &ObjRef<object::PipelineLayout>,
    sets: &[&FinishedSet],
) {
    let dynamic_offsets = sets
        .iter()
        .flat_map(|s| &s.dynamic_offsets)
        .copied()
        .collect::<SmallVec<[_; 16]>>();
    let raw_sets = sets.iter().map(|s| s.set).collect::<SmallVec<[_; 16]>>();

    device.cmd_bind_descriptor_sets(
        cmd,
        bind_point,
        layout.get_handle(),
        0,
        &raw_sets,
        &dynamic_offsets,
    );
}
