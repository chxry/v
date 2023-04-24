use std::ptr;
use std::ffi::{CString, CStr, c_void, c_char};
use ash::{Entry, Device, vk};
use ash::extensions::{ext, khr};
use glfw::{WindowMode, WindowHint, ClientApiHint};
use log::{LevelFilter, Level, debug, log};
use macros::include_glsl;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const FRAMES_IN_FLIGHT: u32 = 2;

fn main() -> Result {
  ezlogger::init(LevelFilter::Trace)?;
  let mut glfw = glfw::init(glfw::LOG_ERRORS)?;
  glfw.window_hint(WindowHint::ClientApi(ClientApiHint::NoApi));
  let (window, _) = glfw
    .create_window(800, 600, "v", WindowMode::Windowed)
    .unwrap();

  unsafe {
    let entry = Entry::load()?;
    let mut extensions = glfw.get_required_instance_extensions().unwrap();
    extensions.push("VK_EXT_debug_utils".to_string());
    extensions.push("VK_KHR_portability_enumeration".to_string());
    debug!("extensions: {:?}", extensions);
    let mut debug_utils_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
      .message_severity(
        // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
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
    let layers = vec![
      "VK_LAYER_KHRONOS_validation",
      // "VK_LAYER_RENDERDOC_Capture",
    ];
    debug!("validation layers: {:?}", layers);
    let instance = entry.create_instance(
      &vk::InstanceCreateInfo::builder()
        .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR)
        .application_info(
          &vk::ApplicationInfo::builder()
            .application_name(&CString::new(env!("CARGO_PKG_NAME"))?)
            .api_version(vk::API_VERSION_1_3),
        )
        .enabled_extension_names(&cstr_vec(extensions))
        .enabled_layer_names(&cstr_vec(layers))
        .push_next(&mut debug_utils_info),
      None,
    )?;
    let debug_utils = ext::DebugUtils::new(&entry, &instance);
    debug_utils.create_debug_utils_messenger(&debug_utils_info, None)?;

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
    let device_extensions = vec!["VK_KHR_swapchain", "VK_KHR_portability_subset"];
    debug!("device extensions: {:?}", device_extensions);
    let device = instance.create_device(
      phys_device,
      &vk::DeviceCreateInfo::builder()
        .queue_create_infos(&[*vk::DeviceQueueCreateInfo::builder()
          .queue_family_index(0)
          .queue_priorities(&[1.0])])
        .enabled_extension_names(&cstr_vec(device_extensions)),
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
        .attachments(&[*vk::AttachmentDescription::builder()
          .format(SWAPCHAIN_FORMAT)
          .samples(vk::SampleCountFlags::TYPE_1)
          .load_op(vk::AttachmentLoadOp::CLEAR)
          .store_op(vk::AttachmentStoreOp::STORE)
          .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
          .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
          .initial_layout(vk::ImageLayout::UNDEFINED)
          .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)])
        .subpasses(&[*vk::SubpassDescription::builder()
          .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
          .color_attachments(&[*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)])])
        .dependencies(&[*vk::SubpassDependency::builder()
          .src_subpass(vk::SUBPASS_EXTERNAL)
          .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
          .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)]),
      None,
    )?;
    let pipeline_layout =
      device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder(), None)?;
    let pipeline = device
      .create_graphics_pipelines(
        vk::PipelineCache::null(),
        &[*vk::GraphicsPipelineCreateInfo::builder()
          .stages(&[*vert_stage, *frag_stage])
          .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::builder())
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
              .cull_mode(vk::CullModeFlags::BACK)
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
      &device,
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
    let resources: Vec<_> = command_buffers
      .into_iter()
      .map(|b| {
        (
          b,
          device
            .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
            .unwrap(),
          device
            .create_semaphore(&vk::SemaphoreCreateInfo::builder(), None)
            .unwrap(),
          device
            .create_fence(
              &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
              None,
            )
            .unwrap(),
        )
      })
      .collect();

    let mut frame: u32 = 0;
    while !window.should_close() {
      glfw.poll_events();

      let resources = resources[frame as usize];
      device.wait_for_fences(&[resources.3], true, u64::MAX)?;
      device.reset_fences(&[resources.3])?;
      let image = swapchain_ext
        .acquire_next_image(swapchain, u64::MAX, resources.1, vk::Fence::null())?
        .0;

      device.reset_command_buffer(resources.0, vk::CommandBufferResetFlags::empty())?;
      device.begin_command_buffer(resources.0, &vk::CommandBufferBeginInfo::builder())?;
      device.cmd_begin_render_pass(
        resources.0,
        &vk::RenderPassBeginInfo::builder()
          .render_pass(render_pass)
          .framebuffer(framebuffers[image as usize])
          .render_area(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: surface_capabilities.current_extent,
          })
          .clear_values(&[vk::ClearValue {
            color: vk::ClearColorValue {
              float32: [0.0, 0.0, 0.0, 1.0],
            },
          }]),
        vk::SubpassContents::INLINE,
      );

      device.cmd_set_viewport(
        resources.0,
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
        resources.0,
        0,
        &[vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: surface_capabilities.current_extent,
        }],
      );
      device.cmd_bind_pipeline(resources.0, vk::PipelineBindPoint::GRAPHICS, pipeline);
      device.cmd_draw(resources.0, 3, 1, 0, 0);

      device.cmd_end_render_pass(resources.0);
      device.end_command_buffer(resources.0)?;

      device.queue_submit(
        queue,
        &[*vk::SubmitInfo::builder()
          .wait_semaphores(&[resources.1])
          .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
          .command_buffers(&[resources.0])
          .signal_semaphores(&[resources.2])],
        resources.3,
      )?;

      match swapchain_ext.queue_present(
        queue,
        &vk::PresentInfoKHR::builder()
          .wait_semaphores(&[resources.2])
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
            &device,
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
  device: &Device,
  swapchain_ext: &khr::Swapchain,
  surface: vk::SurfaceKHR,
  render_pass: vk::RenderPass,
  surface_capabilities: vk::SurfaceCapabilitiesKHR,
) -> Result<(vk::SwapchainKHR, Vec<vk::ImageView>, Vec<vk::Framebuffer>)> {
  let image_count = surface_capabilities.min_image_count + 1;
  let extent = surface_capabilities.current_extent;
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
            .attachments(&[*v])
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

fn cstr_vec<S: Into<Vec<u8>>>(v: Vec<S>) -> Vec<*const c_char> {
  v.into_iter()
    .map(|s| CString::new(s).unwrap().into_raw() as _)
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
