use std::{
    collections::{hash_map::RandomState, HashSet},
    ffi::CStr,
    fmt::Debug,
    sync::Arc,
};

use pumice::{
    loader::{
        tables::{EntryTable, InstanceTable},
        EntryLoader, InstanceLoader,
    },
    util::ApiLoadConfig,
    vk, InstanceWrapper, VulkanResult,
};
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use crate::{
    object,
    tracing::shim_macros::info,
    util::{
        self,
        debug_callback::to_version,
        format_utils::{Fun, IterDisplay},
    },
};

pub(crate) struct InnerInstance {
    pub(crate) entry: pumice::EntryWrapper,
    pub(crate) instance: pumice::InstanceWrapper,

    pub(crate) physical_devices: Vec<vk::PhysicalDevice>,
    pub(crate) physical_device_properties: Vec<vk::PhysicalDeviceProperties>,

    pub(crate) debug_messenger: Option<pumice::extensions::ext_debug_utils::DebugUtilsMessengerEXT>,
    pub(crate) verbose: bool,

    pub(crate) allocation_callbacks: Option<vk::AllocationCallbacks>,

    pub(crate) instance_table: Box<InstanceTable>,
    pub(crate) entry_table: Box<EntryTable>,
    // at the bottom so that it is dropped last, the loader keeps the vulkan dll loaded
    pub(crate) entry_loader: EntryLoader,
}

pub struct InstanceCreateInfo<'a, 'b> {
    pub config: &'a mut ApiLoadConfig<'b>,
    pub validation_layers: &'a [&'a CStr],
    pub enable_debug_callback: bool,
    pub app_name: &'a CStr,
    pub verbose: bool,
}

#[derive(Clone)]
pub struct Instance(Arc<InnerInstance>);

impl Instance {
    pub unsafe fn new(info: InstanceCreateInfo) -> Self {
        let allocation_callbacks = None;

        let InstanceCreateInfo {
            config,
            validation_layers,
            enable_debug_callback,
            app_name,
            verbose,
        } = info;

        if enable_debug_callback {
            config
                .add_extension(pumice::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        let (entry, entry_table, entry_loader) = {
            let mut table = Box::new(EntryTable::new_empty());
            let loader = EntryLoader::new().expect("Failed to create entry loader");
            table.load(&loader);
            let entry = pumice::EntryWrapper::new(&*table);

            (entry, table, loader)
        };

        // api version check
        {
            let api_version = entry
                .enumerate_instance_version()
                .expect("enumerate_instance_version error");

            if api_version < config.get_api_version() {
                let requested = to_version(config.get_api_version());
                let api_version = to_version(api_version);
                panic!(
                    "Unsupported api version, requested {}.{}.{}, available {}.{}.{}",
                    requested.0,
                    requested.1,
                    requested.2,
                    api_version.0,
                    api_version.1,
                    api_version.2
                );
            }
        }

        // instance extension check
        {
            let (available_instance_extensions, _) = entry
                .enumerate_instance_extension_properties(None, None)
                .expect("enumerate_instance_extension_properties error");

            let available_instance_extensions: HashSet<&CStr, RandomState> = HashSet::from_iter(
                available_instance_extensions
                    .iter()
                    .map(|p| CStr::from_ptr(p.extension_name.as_ptr())),
            );

            let missing_extensions = config
                .get_instance_extensions_iter()
                .filter(|e| !available_instance_extensions.contains(e));

            if missing_extensions.clone().next().is_some() {
                let missing =
                    IterDisplay::new(missing_extensions, |i, d| i.to_string_lossy().fmt(d));
                panic!("Instance doesn't support requested extensions:\n{missing}");
            }
        }

        // instance layer check
        {
            let (available_layers, _) = entry
                .enumerate_instance_layer_properties(None)
                .expect("enumerate_instance_extension_properties error");

            let available_layers: HashSet<&CStr, RandomState> = HashSet::from_iter(
                available_layers
                    .iter()
                    .map(|p| CStr::from_ptr(p.layer_name.as_ptr())),
            );

            let missing_extensions = validation_layers
                .iter()
                .filter(|&&e| !available_layers.contains(e));

            if missing_extensions.clone().next().is_some() {
                let missing =
                    IterDisplay::new(missing_extensions, |i, d| i.to_string_lossy().fmt(d));
                panic!("Instance doesn't support requested layers:\n{missing}");
            }
        }

        let (instance_layer_properties, _) = entry
            .enumerate_instance_layer_properties(None)
            .expect("enumerate_instance_layer_properties error");

        if verbose {
            let iter = instance_layer_properties
                .iter()
                .map(|l| CStr::from_ptr(l.layer_name.as_ptr()).to_string_lossy());
            let _iter = IterDisplay::new(iter, |i, w| i.fmt(w));
            info!(
                "{} instance layers:\n{}",
                instance_layer_properties.len(),
                iter
            );
        } else {
            info!("{} instance layers", instance_layer_properties.len());
        }

        let app_info = pumice::vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: 0,
            p_engine_name: app_name.as_ptr(),
            engine_version: 0,
            api_version: config.get_api_version(),
            ..Default::default()
        };

        let instance_extensions = config.get_instance_extensions();

        let validation_layers = validation_layers
            .iter()
            .map(|l| l.as_ptr())
            .collect::<Vec<_>>();

        let create_info = pumice::vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: validation_layers.len() as _,
            pp_enabled_layer_names: validation_layers.as_ptr(),
            enabled_extension_count: instance_extensions.len() as _,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            ..Default::default()
        };

        let (instance, instance_table) = {
            let raw = entry
                .create_instance(&create_info, allocation_callbacks.as_ref())
                .expect("create_instance error");
            let loader = InstanceLoader::new(raw, &entry_loader);
            let mut table = Box::new(InstanceTable::new_empty());
            table.load(&loader, config);
            let instance = InstanceWrapper::new(raw, &*table);

            (instance, table)
        };

        let debug_messenger = if enable_debug_callback {
            let info = pumice::extensions::ext_debug_utils::DebugUtilsMessengerCreateInfoEXT {
                message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::all(),
                message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                pfn_user_callback: Some(util::debug_callback::debug_callback),
                ..Default::default()
            };

            Some(
                instance
                    .create_debug_utils_messenger_ext(&info, allocation_callbacks.as_ref())
                    .expect("create_debug_utils_messenger_ext error"),
            )
        } else {
            None
        };

        let (physical_devices, _) = instance
            .enumerate_physical_devices(None)
            .expect("enumerate_physical_devices error");
        let physical_device_properties = physical_devices
            .iter()
            .map(|&d| instance.get_physical_device_properties(d))
            .collect::<Vec<_>>();

        if verbose {
            let _fun = Fun::new(|f: &mut std::fmt::Formatter<'_>| {
                for (i, property) in physical_device_properties.iter().enumerate() {
                    let name = CStr::from_ptr(property.device_name.as_ptr()).to_string_lossy();
                    let core = to_version(property.api_version);
                    let driver = to_version(property.driver_version);
                    write! {
                        f,
                        "\n[{i}] {name}\n    core {}.{}.{}, driver {}.{}.{}, {:?}",
                        core.0, core.1, core.2,
                        driver.0, driver.1, driver.2,
                        property.device_type,
                    }?;
                }
                Ok(())
            });
            info!("{} physical devices:{}", physical_devices.len(), fun);
        } else {
            info!("{} physical devices", physical_devices.len());
        }

        let inner = InnerInstance {
            entry,
            instance,
            physical_devices,
            physical_device_properties,
            debug_messenger,
            verbose,
            allocation_callbacks,
            instance_table,
            entry_table,
            entry_loader,
        };

        Self(Arc::new(inner))
    }
    pub(crate) fn inner(&self) -> &InnerInstance {
        &self.0
    }
    pub fn handle(&self) -> &InstanceWrapper {
        &self.0.instance
    }
    pub unsafe fn instance_loader(&self) -> InstanceLoader {
        InstanceLoader::new(self.0.instance.handle(), &self.0.entry_loader)
    }
    pub fn allocator_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.0.allocation_callbacks.as_ref()
    }
    pub unsafe fn create_surface<W: HasRawDisplayHandle + HasRawWindowHandle>(
        &self,
        window: &W,
    ) -> VulkanResult<object::Surface> {
        self.handle()
            .create_surface(window, self.allocator_callbacks())
            .map(|raw| object::Surface::from_raw(raw, self.clone()))
    }
}
