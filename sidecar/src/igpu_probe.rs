//! iGPU probe (ticket 03): enumerate every Vulkan device reachable from this
//! OS, report compute capability per device, and emit a machine-readable
//! verdict input. The UHD 730 lives on the Windows side of this WSL2 host;
//! whether it is reachable from Linux at all is exactly what this measures.
//! Uses ash with the "loaded" feature (dlopens libvulkan.so.1 at runtime; no
//! Vulkan SDK needed to build).
use ash::vk;
use serde_json::json;

fn main() {
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("igpu-probe: cannot load Vulkan loader: {e}");
            std::process::exit(2);
        }
    };
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"laya-igpu-probe")
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = match unsafe { entry.create_instance(&create_info, None) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("igpu-probe: vkCreateInstance failed: {e}");
            std::process::exit(2);
        }
    };
    let devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("igpu-probe: vkEnumeratePhysicalDevices failed: {e}");
            std::process::exit(2);
        }
    };
    let mut out = Vec::new();
    for dev in devices.iter().copied() {
        let props = unsafe { instance.get_physical_device_properties(dev) };
        let queues = unsafe { instance.get_physical_device_queue_family_properties(dev) };
        let compute_queues = queues
            .iter()
            .filter(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .count();
        let memory = unsafe { instance.get_physical_device_memory_properties(dev) };
        let heaps: Vec<serde_json::Value> = memory
            .memory_heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| json!({"device_local_bytes": h.size}))
            .collect();
        out.push(json!({
            "name": unsafe { core::ffi::CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy(),
            "type": format!("{:?}", props.device_type),
            "api_version": format!(
                "{}.{}.{}",
                vk::api_version_major(props.api_version),
                vk::api_version_minor(props.api_version),
                vk::api_version_patch(props.api_version)
            ),
            "compute_queue_families": compute_queues,
            "device_local_heaps": heaps,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "vulkan_devices": out,
            "count": devices.len(),
        }))
        .unwrap()
    );
    let _ = unsafe { instance.destroy_instance(None) };
}
