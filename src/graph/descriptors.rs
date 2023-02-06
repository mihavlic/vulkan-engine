use std::{
    num::NonZeroUsize,
    ops::{Deref, DerefMut, Range},
    ptr::NonNull,
};

use bumpalo::Bump;
use pumice::{util::ObjectHandle, vk, DeviceWrapper, VulkanResult};
use smallvec::SmallVec;

use crate::{
    device::Device,
    graph::allocator::round_up_pow2_usize,
    object::{self, DescriptorBinding, ObjRef},
};

use super::execute::GraphExecutor;

struct UniformBuffer {
    buffer: vk::Buffer,
    allocation: pumice_vma::Allocation,
    start: NonNull<u8>,
    cursor: NonNull<u8>,
    end: NonNull<u8>,
}

pub struct DescriptorAllocator {
    free_buffers: Vec<UniformBuffer>,
    free_pools: Vec<vk::DescriptorPool>,
    buffers: Vec<UniformBuffer>,
    pools: Vec<vk::DescriptorPool>,
}

macro_rules! desc_set_sizes {
    ($($count:literal * $kind:ident,)+) => {
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

const UNIFORM_BUFFER_SIZE: u64 = 1024 * 16;

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

impl DescriptorAllocator {
    pub(crate) fn new() -> Self {
        Self {
            free_buffers: Vec::new(),
            free_pools: Vec::new(),
            buffers: Vec::new(),
            pools: Vec::new(),
        }
    }
    pub(crate) unsafe fn reset(&mut self, device: &Device) {
        self.free_buffers
            .extend(self.buffers.drain(..).map(|mut b| {
                b.cursor = b.start;
                b
            }));
        self.free_pools.extend(self.pools.drain(..).map(|p| {
            device.device().reset_descriptor_pool(p, None);
            p
        }));
    }
    pub(crate) unsafe fn destroy(&mut self, device: &Device) {
        for &pool in &self.pools {
            device
                .device()
                .destroy_descriptor_pool(pool, device.allocator_callbacks());
        }
        for buffer in &self.buffers {
            device
                .allocator()
                .destroy_buffer(buffer.buffer, buffer.allocation)
        }
    }
    pub(crate) unsafe fn allocate_set(
        &mut self,
        device: &Device,
        layout: &ObjRef<object::DescriptorSetLayout>,
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
            pool_size_count: DESCRIPTOR_SET_SIZES.len() as u32,
            p_pool_sizes: DESCRIPTOR_SET_SIZES.as_ptr(),
            ..Default::default()
        };
        let pool = device
            .device()
            .create_descriptor_pool(&info, device.allocator_callbacks())
            .unwrap();
        self.pools.push(pool);
    }
    unsafe fn add_buffer(&mut self, device: &Device) {
        if let Some(free) = self.free_buffers.pop() {
            self.buffers.push(free);
            return;
        }

        let buffer_info = vk::BufferCreateInfo {
            flags: vk::BufferCreateFlags::empty(),
            size: UNIFORM_BUFFER_SIZE,
            usage: vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let allocation_info = pumice_vma::AllocationCreateInfo {
            flags: pumice_vma::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                | pumice_vma::AllocationCreateFlags::MAPPED,
            usage: pumice_vma::MemoryUsage::Auto,
            ..Default::default()
        };

        let (buffer, allocation, info) = device
            .allocator()
            .create_buffer(&buffer_info, &allocation_info)
            .unwrap();

        let start: NonNull<u8> = NonNull::new(info.mapped_data.cast()).unwrap();
        let end = NonNull::new(start.as_ptr().add(info.size.try_into().unwrap())).unwrap();

        self.buffers.push(UniformBuffer {
            buffer,
            allocation,
            start,
            cursor: start,
            end,
        });
    }
    pub unsafe fn allocate_uniform_iter<T, I: IntoIterator<Item = T>>(
        &mut self,
        device: &Device,
        iter: I,
    ) -> UniformResult
    where
        I::IntoIter: ExactSizeIterator,
    {
        let iter = iter.into_iter();
        assert!(iter.len() > 0);

        let layout = std::alloc::Layout::array::<T>(iter.len()).unwrap();
        let uniform = self.allocate_uniform_raw(device, layout);

        let UniformMemory {
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
        device: &Device,
        value: T,
    ) -> UniformResult {
        let layout = std::alloc::Layout::new::<T>();
        let uniform = self.allocate_uniform_raw(device, layout);

        let UniformMemory {
            dynamic_offset,
            buffer,
            memory,
        } = uniform;
        let ptr = memory.as_ptr().cast::<T>();
        *ptr = value;

        UniformResult {
            dynamic_offset,
            buffer,
        }
    }
    pub(crate) unsafe fn allocate_uniform_raw(
        &mut self,
        device: &Device,
        layout: std::alloc::Layout,
    ) -> UniformMemory {
        assert!(layout.size() as u64 <= UNIFORM_BUFFER_SIZE);

        if self.buffers.is_empty() {
            self.add_buffer(device);
        }

        let buffer = self.buffers.last_mut().unwrap();
        let ptr = Self::bump_buffer(buffer, layout, device).unwrap_or_else(|| {
            self.add_buffer(device);
            let buffer = self.buffers.last_mut().unwrap();
            Self::bump_buffer(buffer, layout, device)
                .expect("Failed to bump allocate from a fresh buffer")
        });

        let buffer = self.buffers.last().unwrap();

        UniformMemory {
            dynamic_offset: ptr
                .as_ptr()
                .offset_from(buffer.start.as_ptr())
                .try_into()
                .unwrap(),
            buffer: buffer.buffer,
            memory: ptr,
        }
    }
    unsafe fn bump_buffer(
        buffer: &mut UniformBuffer,
        layout: std::alloc::Layout,
        device: &Device,
    ) -> Option<NonNull<u8>> {
        let start = buffer.cursor.as_ptr() as usize;
        let aligned = round_up_pow2_usize(start, layout.align());
        let offset = aligned - start;

        let start_ptr = buffer.cursor.as_ptr().add(offset);
        let end_ptr = start_ptr.add(layout.size());

        if end_ptr > buffer.end.as_ptr() {
            return None;
        }

        buffer.cursor = NonNull::new(end_ptr).unwrap();

        Some(NonNull::new(start_ptr).unwrap())
    }
}

pub struct UniformMemory {
    pub dynamic_offset: u32,
    pub buffer: vk::Buffer,
    pub memory: NonNull<u8>,
}

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

#[derive(Default, Clone)]
pub struct DescBuffer {
    pub buffer: vk::Buffer,
    pub view: vk::BufferView,
    pub offset: u64,
    pub range: u64,
    pub dynamic_offset: Option<u32>,
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
    pub fn update_image_binding(&mut self, binding: u32, array_offset: u32, data: &DescImage) {
        self.update_image_bindings_arr(binding, array_offset, std::slice::from_ref(data));
    }
    pub fn update_buffer_binding(&mut self, binding: u32, array_offset: u32, data: &DescBuffer) {
        self.update_buffer_bindings_arr(binding, array_offset, std::slice::from_ref(data));
    }
    pub fn update_image_bindings_arr(
        &mut self,
        binding: u32,
        array_offset: u32,
        datas: &[DescImage],
    ) {
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
    }
    pub fn update_buffer_bindings_arr(
        &mut self,
        binding: u32,
        array_offset: u32,
        datas: &[DescBuffer],
    ) {
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
    }
    pub fn update_whole(&mut self, bindings: &[DescriptorData]) {
        let layout = &self.layout.get_create_info();
        assert_eq!(bindings.len(), layout.bindings.len());

        let bump = Bump::new();
        for ((desc, data), i) in layout.bindings.iter().zip(bindings).zip(0u32..) {
            let count = desc.count as usize;
            match data {
                DescriptorData::Image(image) => {
                    assert_eq!(count, 1);
                    self.update_image_bindings_arr(i, 0, std::slice::from_ref(image))
                }
                DescriptorData::ImageArr(images) => {
                    assert_eq!(count, images.len());
                    self.update_image_bindings_arr(i, 0, *images)
                }
                DescriptorData::Buffer(buffer) => {
                    assert_eq!(count, 1);
                    self.update_buffer_bindings_arr(i, 0, std::slice::from_ref(buffer))
                }
                DescriptorData::BufferArr(buffers) => {
                    assert_eq!(count, buffers.len());
                    self.update_buffer_bindings_arr(i, 0, *buffers)
                }
            }
        }
    }
    pub unsafe fn finish(&mut self, executor: &GraphExecutor) -> FinishedSet {
        let state = executor.graph.state();
        let mut descriptor_allocator = state.descriptor_allocator.borrow_mut();
        self.finish_internal(&state.device, &mut descriptor_allocator)
    }
    unsafe fn finish_internal(
        &mut self,
        device: &Device,
        allocator: &mut DescriptorAllocator,
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
            self.set = allocator.allocate_set(device, self.layout);
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

pub unsafe fn bind_descriptor_sets(
    device: &DeviceWrapper,
    cmd: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    layout: &ObjRef<object::PipelineLayout>,
    sets: &[FinishedSet],
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
