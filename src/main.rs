#![feature(generic_arg_infer, type_name_of_val)]
use std::{ptr, mem};
use std::marker::PhantomData;
use std::ffi::{CString, CStr, c_void, c_char};
use ash::{Entry, Instance, Device, vk};
use ash::extensions::{ext, khr};
use glfw::{WindowMode, WindowHint, ClientApiHint};
use glam::{Vec3, Vec2, Mat4};
use log::{LevelFilter, Level, debug, log};
use macros::include_glsl;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const FRAMES_IN_FLIGHT: u32 = 2;
const EXTENSIONS: [&str; 2] = [
  #[cfg(debug_assertions)]
  "VK_EXT_debug_utils",
  #[cfg(target_os = "macos")]
  "VK_KHR_portability_enumeration",
];
const DEVICE_EXTENSIONS: [&str; 2] = [
  "VK_KHR_swapchain",
  #[cfg(target_os = "macos")]
  "VK_KHR_portability_subset",
];
const LAYERS: [&str; 1] = [
  #[cfg(debug_assertions)]
  "VK_LAYER_KHRONOS_validation",
];

#[repr(C)]
#[derive(Copy, Clone)]
struct Vertex {
  pos: Vec3,
  uv: Vec2,
}

#[repr(C)]
struct UniformBuffer {
  model: Mat4,
  view: Mat4,
  proj: Mat4,
}

impl Vertex {
  fn bind_desc() -> vk::VertexInputBindingDescription {
    *vk::VertexInputBindingDescription::builder()
      .binding(0)
      .stride(mem::size_of::<Self>() as _)
      .input_rate(vk::VertexInputRate::VERTEX)
  }

  fn attr_desc() -> [vk::VertexInputAttributeDescription; 2] {
    [
      *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0),
      *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(mem::size_of::<Vec3>() as _),
    ]
  }
}

unsafe fn get_memory_type(
  instance: &Instance,
  phys_device: vk::PhysicalDevice,
  props: vk::MemoryPropertyFlags,
) -> u32 {
  instance
    .get_physical_device_memory_properties(phys_device)
    .memory_types
    .iter()
    .position(|t| t.property_flags.contains(props))
    .expect("could not find suitable memory") as _
}

unsafe fn exec_command_buffer<F: Fn(vk::CommandBuffer)>(
  device: &Device,
  queue: vk::Queue,
  command_pool: vk::CommandPool,
  f: F,
) -> Result {
  let command_buffer = device.allocate_command_buffers(
    &vk::CommandBufferAllocateInfo::builder()
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_pool(command_pool)
      .command_buffer_count(1),
  )?[0];
  device.begin_command_buffer(
    command_buffer,
    &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
  )?;
  f(command_buffer);
  device.end_command_buffer(command_buffer)?;
  device.queue_submit(
    queue,
    &[*vk::SubmitInfo::builder().command_buffers(&[command_buffer])],
    vk::Fence::null(),
  )?;
  device.queue_wait_idle(queue)?;
  device.free_command_buffers(command_pool, &[command_buffer]);
  Ok(())
}

// todo make image wrapper
unsafe fn transition_image(
  device: &Device,
  command_buffer: vk::CommandBuffer,
  image: vk::Image,
  old: vk::ImageLayout,
  new: vk::ImageLayout,
) {
  let (old_mask, old_stage) = if old == vk::ImageLayout::TRANSFER_DST_OPTIMAL {
    (
      vk::AccessFlags::TRANSFER_WRITE,
      vk::PipelineStageFlags::TRANSFER,
    )
  } else {
    (vk::AccessFlags::NONE, vk::PipelineStageFlags::TOP_OF_PIPE)
  };
  let (new_mask, new_stage) = if new == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
    (
      vk::AccessFlags::SHADER_READ,
      vk::PipelineStageFlags::FRAGMENT_SHADER,
    )
  } else {
    (
      vk::AccessFlags::TRANSFER_WRITE,
      vk::PipelineStageFlags::TRANSFER,
    )
  };
  device.cmd_pipeline_barrier(
    command_buffer,
    old_stage,
    new_stage,
    vk::DependencyFlags::empty(),
    &[],
    &[],
    &[*vk::ImageMemoryBarrier::builder()
      .old_layout(old)
      .new_layout(new)
      .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
      .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
      .src_access_mask(old_mask)
      .dst_access_mask(new_mask)
      .image(image)
      .subresource_range(
        *vk::ImageSubresourceRange::builder()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_mip_level(0)
          .level_count(1)
          .base_array_layer(0)
          .layer_count(1),
      )],
  );
}

// todo drop
struct Buffer<T, P = ()> {
  buf: vk::Buffer,
  mem: vk::DeviceMemory,
  ptr: P,
  len: usize,
  t: PhantomData<T>,
}

impl<T> Buffer<T, ()> {
  unsafe fn new(
    instance: &Instance,
    device: &Device,
    phys_device: vk::PhysicalDevice,
    usage: vk::BufferUsageFlags,
    mem_props: vk::MemoryPropertyFlags,
    len: usize,
  ) -> Result<Self> {
    let size = (len * mem::size_of::<T>()) as _;
    let buf = device.create_buffer(
      &vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE),
      None,
    )?;
    let mem = device.allocate_memory(
      &vk::MemoryAllocateInfo::builder()
        .allocation_size(device.get_buffer_memory_requirements(buf).size)
        .memory_type_index(get_memory_type(instance, phys_device, mem_props)),
      None,
    )?;
    device.bind_buffer_memory(buf, mem, 0)?;
    Ok(Self {
      buf,
      mem,
      len,
      ptr: (),
      t: PhantomData,
    })
  }

  unsafe fn map(self, device: &Device) -> Result<Buffer<T, *mut T>> {
    Ok(Buffer {
      buf: self.buf,
      mem: self.mem,
      len: self.len,
      ptr: device.map_memory(
        self.mem,
        0,
        (self.len * mem::size_of::<T>()) as _,
        vk::MemoryMapFlags::empty(),
      )? as _,
      t: PhantomData,
    })
  }

  unsafe fn write(&self, device: &Device, data: &[T]) -> Result {
    ptr::copy(
      data.as_ptr(),
      device.map_memory(
        self.mem,
        0,
        (data.len() * mem::size_of::<T>()) as _,
        vk::MemoryMapFlags::empty(),
      )? as _,
      data.len(),
    );
    device.unmap_memory(self.mem);
    Ok(())
  }

  unsafe fn copy(
    &self,
    device: &Device,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    other: &Self,
  ) -> Result {
    exec_command_buffer(device, queue, command_pool, |command_buffer| {
      device.cmd_copy_buffer(
        command_buffer,
        self.buf,
        other.buf,
        &[*vk::BufferCopy::builder().size((self.len * mem::size_of::<T>()) as _)],
      );
    })?;
    Ok(())
  }

  unsafe fn new_with_staging(
    instance: &Instance,
    device: &Device,
    phys_device: vk::PhysicalDevice,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    usage: vk::BufferUsageFlags,
    data: &[T],
  ) -> Result<Self> {
    let staging = Self::new(
      instance,
      device,
      phys_device,
      vk::BufferUsageFlags::TRANSFER_SRC,
      vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
      data.len(),
    )?;
    staging.write(device, data)?;
    let buf = Self::new(
      instance,
      device,
      phys_device,
      usage | vk::BufferUsageFlags::TRANSFER_DST,
      vk::MemoryPropertyFlags::DEVICE_LOCAL,
      data.len(),
    )?;
    staging.copy(device, queue, command_pool, &buf)?;
    Ok(buf)
  }
}

impl<T> Buffer<T, *mut T> {
  unsafe fn write(&self, data: &[T]) {
    ptr::copy(data.as_ptr(), self.ptr, data.len());
  }
}

struct FrameResources {
  command_buffer: vk::CommandBuffer,
  descriptor_sets: [vk::DescriptorSet; 2],
  image_availible: vk::Semaphore,
  render_finished: vk::Semaphore,
  in_flight: vk::Fence,
  uniform_buffer: Buffer<UniformBuffer, *mut UniformBuffer>,
}

fn main() -> Result {
  ezlogger::init(LevelFilter::Trace)?;
  let mut glfw = glfw::init(glfw::LOG_ERRORS)?;
  glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
  let (window, _) = glfw
    .create_window(800, 600, "v", WindowMode::Windowed)
    .unwrap();

  unsafe {
    let entry = Entry::load()?;
    let extensions = glfw.get_required_instance_extensions().unwrap();
    let mut extensions: Vec<_> = extensions.iter().map(String::as_str).collect();
    extensions.extend(EXTENSIONS);
    debug!("extensions: {:?}", extensions);
    let extensions = cstr_vec(&extensions);

    let app_name = CString::new(env!("CARGO_PKG_NAME"))?;
    let app_info = vk::ApplicationInfo::builder()
      .application_name(&app_name)
      .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::builder()
      .flags(if cfg!(target_os = "macos") {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
      } else {
        vk::InstanceCreateFlags::empty()
      })
      .application_info(&app_info)
      .enabled_extension_names(&extensions);
    let instance = if cfg!(debug_assertions) {
      let mut debug_utils_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
          vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
          },
        )
        .message_type(
          vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));
      debug!("validation layers: {:?}", LAYERS);
      let instance = entry.create_instance(
        &instance_info
          .enabled_layer_names(&cstr_vec(&LAYERS))
          .push_next(&mut debug_utils_info),
        None,
      )?;
      let debug_utils = ext::DebugUtils::new(&entry, &instance);
      debug_utils.create_debug_utils_messenger(&debug_utils_info, None)?;
      instance
    } else {
      entry.create_instance(&instance_info, None)?
    };

    let surface_ext = khr::Surface::new(&entry, &instance);
    let mut surface = vk::SurfaceKHR::null();
    window
      .create_window_surface(instance.handle(), ptr::null(), &mut surface)
      .result()?;

    let phys_device = instance.enumerate_physical_devices()?[0];
    let phys_device_props = instance.get_physical_device_properties(phys_device);
    debug!(
      "device: {:?}",
      CStr::from_ptr(phys_device_props.device_name.as_ptr())
    );
    debug!("device extensions: {:?}", DEVICE_EXTENSIONS);
    let device = instance.create_device(
      phys_device,
      &vk::DeviceCreateInfo::builder()
        .queue_create_infos(&[*vk::DeviceQueueCreateInfo::builder()
          .queue_family_index(0)
          .queue_priorities(&[1.0])])
        .enabled_extension_names(&cstr_vec(&DEVICE_EXTENSIONS)),
      None,
    )?;
    let queue = device.get_device_queue(0, 0);

    let main = CString::new("main")?;
    let vert_module = device.create_shader_module(
      &vk::ShaderModuleCreateInfo::builder().code(&include_glsl!("shaders/shader.vert")),
      None,
    )?;
    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
      .module(vert_module)
      .stage(vk::ShaderStageFlags::VERTEX)
      .name(&main);
    let frag_module = device.create_shader_module(
      &vk::ShaderModuleCreateInfo::builder().code(&include_glsl!("shaders/shader.frag")),
      None,
    )?;
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
      .module(frag_module)
      .stage(vk::ShaderStageFlags::FRAGMENT)
      .name(&main);
    let render_pass = device.create_render_pass(
      &vk::RenderPassCreateInfo::builder()
        .attachments(&[
          *vk::AttachmentDescription::builder()
            .format(SWAPCHAIN_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
          *vk::AttachmentDescription::builder()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ])
        .subpasses(&[*vk::SubpassDescription::builder()
          .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
          .color_attachments(&[*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])
          .depth_stencil_attachment(
            &vk::AttachmentReference::builder()
              .attachment(1)
              .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
          )])
        .dependencies(&[*vk::SubpassDependency::builder()
          .src_subpass(vk::SUBPASS_EXTERNAL)
          .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
              | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
          )
          .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
              | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
          )
          .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
              | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
          )]),
      None,
    )?;
    let descriptor_set_layout = device.create_descriptor_set_layout(
      &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
        *vk::DescriptorSetLayoutBinding::builder()
          .binding(0)
          .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
          .descriptor_count(1)
          .stage_flags(vk::ShaderStageFlags::VERTEX),
        *vk::DescriptorSetLayoutBinding::builder()
          .binding(1)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .descriptor_count(1)
          .stage_flags(vk::ShaderStageFlags::FRAGMENT),
      ]),
      None,
    )?;
    let pipeline_layout = device.create_pipeline_layout(
      &vk::PipelineLayoutCreateInfo::builder().set_layouts(&[descriptor_set_layout]),
      None,
    )?;
    let pipeline = device
      .create_graphics_pipelines(
        vk::PipelineCache::null(),
        &[*vk::GraphicsPipelineCreateInfo::builder()
          .stages(&[*vert_stage, *frag_stage])
          .vertex_input_state(
            &vk::PipelineVertexInputStateCreateInfo::builder()
              .vertex_binding_descriptions(&[Vertex::bind_desc()])
              .vertex_attribute_descriptions(&Vertex::attr_desc()),
          )
          .input_assembly_state(
            &vk::PipelineInputAssemblyStateCreateInfo::builder()
              .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
          )
          .viewport_state(
            &vk::PipelineViewportStateCreateInfo::builder()
              .viewport_count(1)
              .scissor_count(1),
          )
          .rasterization_state(
            &vk::PipelineRasterizationStateCreateInfo::builder()
              .depth_clamp_enable(false)
              .depth_bias_enable(false)
              .polygon_mode(vk::PolygonMode::FILL)
              .line_width(1.0)
              .cull_mode(vk::CullModeFlags::NONE) // back
              .front_face(vk::FrontFace::CLOCKWISE),
          )
          .multisample_state(
            &vk::PipelineMultisampleStateCreateInfo::builder()
              .sample_shading_enable(false)
              .rasterization_samples(vk::SampleCountFlags::TYPE_1),
          )
          .color_blend_state(
            &vk::PipelineColorBlendStateCreateInfo::builder()
              .logic_op_enable(false)
              .attachments(&[*vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(false)]),
          )
          .dynamic_state(
            &vk::PipelineDynamicStateCreateInfo::builder()
              .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
          )
          .depth_stencil_state(
            &vk::PipelineDepthStencilStateCreateInfo::builder()
              .depth_test_enable(true)
              .depth_write_enable(true)
              .depth_compare_op(vk::CompareOp::LESS),
          )
          .layout(pipeline_layout)
          .render_pass(render_pass)
          .subpass(0)],
        None,
      )
      .map_err(|e| e.1)?[0];

    let swapchain_ext = khr::Swapchain::new(&instance, &device);
    let mut surface_capabilities =
      surface_ext.get_physical_device_surface_capabilities(phys_device, surface)?;
    let (mut swapchain, mut swapchain_image_views, mut framebuffers) = create_swapchain(
      &instance,
      &device,
      phys_device,
      &swapchain_ext,
      surface,
      render_pass,
      surface_capabilities,
    )?;

    let command_pool = device.create_command_pool(
      &vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(0),
      None,
    )?;
    let command_buffers = device.allocate_command_buffers(
      &vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(FRAMES_IN_FLIGHT),
    )?;
    let descriptor_pool = device.create_descriptor_pool(
      &vk::DescriptorPoolCreateInfo::builder()
        .max_sets(FRAMES_IN_FLIGHT)
        .pool_sizes(&[
          *vk::DescriptorPoolSize::builder()
            .descriptor_count(FRAMES_IN_FLIGHT)
            .ty(vk::DescriptorType::UNIFORM_BUFFER),
          *vk::DescriptorPoolSize::builder()
            .descriptor_count(FRAMES_IN_FLIGHT)
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ]),
      None,
    )?;
    let descriptor_sets = device.allocate_descriptor_sets(
      &vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&[descriptor_set_layout; FRAMES_IN_FLIGHT as _]),
    )?;

    let (doc, buffers, images) = gltf::import("model.glb")?;
    let mesh = doc.meshes().nth(1).unwrap();
    let primitive = mesh.primitives().next().unwrap();
    let reader = primitive.reader(|b| Some(&buffers[b.index()]));
    let vertices: Vec<_> = reader
      .read_positions()
      .unwrap()
      .zip(reader.read_tex_coords(0).unwrap().into_f32())
      .map(|(p, u)| Vertex {
        pos: p.into(),
        uv: u.into(),
      })
      .collect();
    let indices: Vec<_> = reader.read_indices().unwrap().into_u32().collect();

    let vertex_buffer = Buffer::new_with_staging(
      &instance,
      &device,
      phys_device,
      queue,
      command_pool,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &vertices,
    )?;
    let index_buffer = Buffer::new_with_staging(
      &instance,
      &device,
      phys_device,
      queue,
      command_pool,
      vk::BufferUsageFlags::INDEX_BUFFER,
      &indices,
    )?;

    let image_format = vk::Format::R8G8B8A8_SRGB;
    let image_data = &images[primitive
      .material()
      .pbr_metallic_roughness()
      .base_color_texture()
      .unwrap()
      .texture()
      .source()
      .index()];
    let image_pixels: Vec<_> = image_data
      .pixels
      .chunks_exact(3)
      .flat_map(|a| [a[0], a[1], a[2], 255])
      .collect();

    let image_staging = Buffer::new(
      &instance,
      &device,
      phys_device,
      vk::BufferUsageFlags::TRANSFER_SRC,
      vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
      image_pixels.len(),
    )?;
    image_staging.write(&device, &image_pixels)?;
    let image_extent = vk::Extent3D {
      width: image_data.width,
      height: image_data.height,
      depth: 1,
    };
    let image = device.create_image(
      &vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(image_extent)
        .mip_levels(1)
        .array_layers(1)
        .format(image_format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1),
      None,
    )?;
    let image_memory = device.allocate_memory(
      &vk::MemoryAllocateInfo::builder()
        .allocation_size(device.get_image_memory_requirements(image).size)
        .memory_type_index(get_memory_type(
          &instance,
          phys_device,
          vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )),
      None,
    )?;
    device.bind_image_memory(image, image_memory, 0)?;
    exec_command_buffer(&device, queue, command_pool, |command_buffer| {
      transition_image(
        &device,
        command_buffer,
        image,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
      );
      device.cmd_copy_buffer_to_image(
        command_buffer,
        image_staging.buf,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[*vk::BufferImageCopy::builder()
          .buffer_offset(0)
          .buffer_row_length(0)
          .buffer_image_height(0)
          .image_subresource(
            *vk::ImageSubresourceLayers::builder()
              .aspect_mask(vk::ImageAspectFlags::COLOR)
              .mip_level(0)
              .base_array_layer(0)
              .layer_count(1),
          )
          .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
          .image_extent(image_extent)],
      );
      transition_image(
        &device,
        command_buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
      );
    })?;
    let image_view = device.create_image_view(
      &vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(image_format)
        .subresource_range(
          *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        ),
      None,
    )?;
    let image_sampler = device.create_sampler(
      &vk::SamplerCreateInfo::builder()
        .min_filter(vk::Filter::LINEAR)
        .mag_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT),
      None,
    )?;

    let resources: Vec<_> = command_buffers
      .into_iter()
      .zip(descriptor_sets)
      .map(|(b, d)| {
        let uniform_buffer = Buffer::new(
          &instance,
          &device,
          phys_device,
          vk::BufferUsageFlags::UNIFORM_BUFFER,
          vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
          1,
        )
        .unwrap()
        .map(&device)
        .unwrap();
        device.update_descriptor_sets(
          &[
            *vk::WriteDescriptorSet::builder()
              .dst_set(d)
              .dst_binding(0)
              .dst_array_element(0)
              .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
              .buffer_info(&[*vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffer.buf)
                .offset(0)
                .range(mem::size_of::<UniformBuffer>() as _)]),
            *vk::WriteDescriptorSet::builder()
              .dst_set(d)
              .dst_binding(1)
              .dst_array_element(0)
              .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
              .image_info(&[*vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image_view)
                .sampler(image_sampler)]),
          ],
          &[],
        );
        FrameResources {
          command_buffer: b,
          descriptor_sets: [d, d.clone()],
          image_availible: device
            .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
            .unwrap(),
          render_finished: device
            .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
            .unwrap(),
          in_flight: device
            .create_fence(
              &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
              None,
            )
            .unwrap(),
          uniform_buffer,
        }
      })
      .collect();

    let mut frame: u32 = 0;
    while !window.should_close() {
      glfw.poll_events();

      let resources = &resources[frame as usize];
      device.wait_for_fences(&[resources.in_flight], true, u64::MAX)?;
      device.reset_fences(&[resources.in_flight])?;
      let image = swapchain_ext
        .acquire_next_image(
          swapchain,
          u64::MAX,
          resources.image_availible,
          vk::Fence::null(),
        )?
        .0;

      device.reset_command_buffer(
        resources.command_buffer,
        vk::CommandBufferResetFlags::empty(),
      )?;
      device.begin_command_buffer(
        resources.command_buffer,
        &vk::CommandBufferBeginInfo::builder(),
      )?;
      device.cmd_begin_render_pass(
        resources.command_buffer,
        &vk::RenderPassBeginInfo::builder()
          .render_pass(render_pass)
          .framebuffer(framebuffers[image as usize])
          .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: surface_capabilities.current_extent,
          })
          .clear_values(&[
            vk::ClearValue {
              color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
              },
            },
            vk::ClearValue {
              depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
              },
            },
          ]),
        vk::SubpassContents::INLINE,
      );

      device.cmd_set_viewport(
        resources.command_buffer,
        0,
        &[vk::Viewport {
          x: 0.0,
          y: 0.0,
          width: surface_capabilities.current_extent.width as _,
          height: surface_capabilities.current_extent.height as _,
          min_depth: 0.0,
          max_depth: 1.0,
        }],
      );
      device.cmd_set_scissor(
        resources.command_buffer,
        0,
        &[vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: surface_capabilities.current_extent,
        }],
      );
      device.cmd_bind_pipeline(
        resources.command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline,
      );

      resources.uniform_buffer.write(&[UniformBuffer {
        model: Mat4::from_rotation_z(glfw.get_time() as _),
        view: Mat4::look_at_rh(Vec3::new(4.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 0.0), Vec3::Z),
        proj: Mat4::perspective_rh(
          80.0f32.to_radians(),
          surface_capabilities.current_extent.width as f32
            / surface_capabilities.current_extent.height as f32,
          0.1,
          100.0,
        ),
      }]);
      device.cmd_bind_vertex_buffers(resources.command_buffer, 0, &[vertex_buffer.buf], &[0]);
      device.cmd_bind_index_buffer(
        resources.command_buffer,
        index_buffer.buf,
        0,
        vk::IndexType::UINT32,
      );
      device.cmd_bind_descriptor_sets(
        resources.command_buffer,
        vk::PipelineBindPoint::GRAPHICS,
        pipeline_layout,
        0,
        &[resources.descriptor_sets[0]],
        &[],
      );
      device.cmd_draw_indexed(resources.command_buffer, indices.len() as _, 1, 0, 0, 0);

      device.cmd_end_render_pass(resources.command_buffer);
      device.end_command_buffer(resources.command_buffer)?;

      device.queue_submit(
        queue,
        &[*vk::SubmitInfo::builder()
          .wait_semaphores(&[resources.image_availible])
          .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
          .command_buffers(&[resources.command_buffer])
          .signal_semaphores(&[resources.render_finished])],
        resources.in_flight,
      )?;

      match swapchain_ext.queue_present(
        queue,
        &vk::PresentInfoKHR::builder()
          .wait_semaphores(&[resources.render_finished])
          .swapchains(&[swapchain])
          .image_indices(&[image]),
      ) {
        Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
          device.device_wait_idle()?;
          for fb in &framebuffers {
            device.destroy_framebuffer(*fb, None);
          }
          for view in &swapchain_image_views {
            device.destroy_image_view(*view, None);
          }
          swapchain_ext.destroy_swapchain(swapchain, None);
          surface_capabilities =
            surface_ext.get_physical_device_surface_capabilities(phys_device, surface)?;
          (swapchain, swapchain_image_views, framebuffers) = create_swapchain(
            &instance,
            &device,
            phys_device,
            &swapchain_ext,
            surface,
            render_pass,
            surface_capabilities,
          )?;
          continue;
        }
        Err(e) => return Err(e.into()),
        _ => {}
      };
      frame = (frame + 1) % FRAMES_IN_FLIGHT;
    }
  }
  Ok(())
}

unsafe fn create_swapchain(
  instance: &Instance,
  device: &Device,
  phys_device: vk::PhysicalDevice,
  swapchain_ext: &khr::Swapchain,
  surface: vk::SurfaceKHR,
  render_pass: vk::RenderPass,
  surface_capabilities: vk::SurfaceCapabilitiesKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::ImageView>, Vec<vk::Framebuffer>)> {
  let image_count = surface_capabilities.min_image_count + 1;
  let extent = surface_capabilities.current_extent;
  // free
  let depth_image = device.create_image(
    &vk::ImageCreateInfo::builder()
      .image_type(vk::ImageType::TYPE_2D)
      .extent(extent.into())
      .mip_levels(1)
      .array_layers(1)
      .format(vk::Format::D32_SFLOAT)
      .tiling(vk::ImageTiling::OPTIMAL)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1),
    None,
  )?;
  let depth_memory = device.allocate_memory(
    &vk::MemoryAllocateInfo::builder()
      .allocation_size(device.get_image_memory_requirements(depth_image).size)
      .memory_type_index(get_memory_type(
        &instance,
        phys_device,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
      )),
    None,
  )?;
  device.bind_image_memory(depth_image, depth_memory, 0)?;
  let depth_view = device.create_image_view(
    &vk::ImageViewCreateInfo::builder()
      .image(depth_image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(vk::Format::D32_SFLOAT)
      .subresource_range(
        *vk::ImageSubresourceRange::builder()
          .aspect_mask(vk::ImageAspectFlags::DEPTH)
          .base_mip_level(0)
          .level_count(1)
          .base_array_layer(0)
          .layer_count(1),
      ),
    None,
  )?;
  let swapchain = swapchain_ext.create_swapchain(
    &vk::SwapchainCreateInfoKHR::builder()
      .surface(surface)
      .min_image_count(if surface_capabilities.max_image_count > 0 {
        image_count.max(surface_capabilities.max_image_count)
      } else {
        image_count
      })
      .image_format(SWAPCHAIN_FORMAT)
      .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
      .image_extent(extent)
      .image_array_layers(1)
      .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .pre_transform(surface_capabilities.current_transform)
      .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
      .present_mode(vk::PresentModeKHR::FIFO)
      .clipped(true),
    None,
  )?;
  let swapchain_image_views: Vec<_> = swapchain_ext
    .get_swapchain_images(swapchain)?
    .iter()
    .map(|i| {
      device
        .create_image_view(
          &vk::ImageViewCreateInfo::builder()
            .image(*i)
            .format(SWAPCHAIN_FORMAT)
            .view_type(vk::ImageViewType::TYPE_2D)
            .subresource_range(
              *vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
            ),
          None,
        )
        .unwrap()
    })
    .collect();
  let framebuffers = swapchain_image_views
    .iter()
    .map(|v| {
      device
        .create_framebuffer(
          &vk::FramebufferCreateInfo::builder()
            .render_pass(render_pass)
            .attachments(&[*v, depth_view])
            .width(extent.width)
            .height(extent.height)
            .layers(1),
          None,
        )
        .unwrap()
    })
    .collect();
  Ok((swapchain, swapchain_image_views, framebuffers))
}

fn cstr_vec(v: &[&str]) -> Vec<*const c_char> {
  v.iter()
    .map(|s| CString::new(*s).unwrap().into_raw() as _)
    .collect::<Vec<_>>()
}

unsafe extern "system" fn debug_callback(
  severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  typ: vk::DebugUtilsMessageTypeFlagsEXT,
  data: *const vk::DebugUtilsMessengerCallbackDataEXT,
  _: *mut c_void,
) -> u32 {
  log!(
    match severity {
      vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
      vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
      vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Debug,
      _ => Level::Trace,
    },
    "({}) {}",
    match typ {
      vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "general",
      vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "validation",
      vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "performance",
      _ => "?",
    },
    CStr::from_ptr((*data).p_message).to_string_lossy()
  );
  0
}
