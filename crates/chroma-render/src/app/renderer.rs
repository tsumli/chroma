use crate::{
    common::{
        self,
        camera::TransformParams,
        command_buffer::create_command_buffers,
        consts::MAX_FRAMES_IN_FLIGHT,
        device::{
            create_logical_device,
            pick_physical_device,
        },
        image_buffer::ImageBuffer,
        layer::{
            check_layer_support,
            required_layer_names,
        },
        sync::SyncObjects,
        uniform_buffer,
    },
    control,
    draw::{
        self,
        strategy::DrawStrategy,
    },
    utils::time::Timer,
};
use anyhow::{
    bail,
    Context as _,
    Result,
};
use ash::vk::{
    self,
    CommandBufferSubmitInfo,
    CommandPoolCreateFlags,
    Extent3D,
    ImageCreateInfo,
    ImageSubresourceRange,
    PipelineStageFlags2,
};
use imgui::{
    FontConfig,
    FontSource,
};
use imgui_winit_support::{
    HiDpiMode,
    WinitPlatform,
};
use winit::{
    dpi::{
        PhysicalPosition,
        PhysicalSize,
    },
    window::Window,
};

pub struct Renderer<'a> {
    graphics_compute_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain: common::swapchain::Swapchain,
    _command_pool: common::command_pool::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    sync_objects: SyncObjects,
    _debug_utils: common::debug::DebugUtils,
    device: common::device::Device,
    _instance: common::instance::Instance,
    _surface: common::surface::Surface,
    timer: Timer,
    output_render_target: ImageBuffer,
    _physical_device: vk::PhysicalDevice,
    output_framebuffers: Vec<common::framebuffer::Framebuffer>,
    imgui_renderer: imgui_rs_vulkan_renderer::Renderer,
    camera: common::camera::Camera,
    draw_strategy: draw::deferred::Deferred<'a>,
    transform_ubo: uniform_buffer::UniformBuffer<TransformParams>,
    camera_ubo: uniform_buffer::UniformBuffer<common::camera::CameraParams>,
    imgui_render_pass: common::render_pass::RenderPass,
}

impl Drop for Renderer<'_> {
    fn drop(&mut self) {
        self.device_wait_idle();
    }
}

impl Renderer<'_> {
    pub fn new(
        window: &Window,
        window_size: PhysicalSize<u32>,
        scene: &chroma_scene::scene::Scene,
        imgui_platform: &mut WinitPlatform,
        imgui_context: &mut imgui::Context,
    ) -> Result<Self> {
        // Init vulkan stuff
        log::info!("loading vulkan entry");
        let entry = unsafe { ash::Entry::load()? };

        log::info!("creating instance");
        let instance = common::instance::Instance::new(&entry)?;
        if check_layer_support(&entry, &required_layer_names())? {
            log::info!("Validation layers are supported!");
        } else {
            bail!("Validation layers are not supported!");
        }

        log::info!("setting up debug utils");
        let debug_utils = common::debug::DebugUtils::new(&entry, &instance.instance())?;

        log::info!("creating surface");
        let surface = common::surface::Surface::new(&entry, &instance.instance(), &window)?;

        log::info!("picking physical device");
        let physical_device = pick_physical_device(&instance.instance(), &surface)?;

        log::info!("Get queue family indices");
        let queue_family_indices =
            common::device::find_queue_family(&instance.instance(), physical_device, &surface)?;

        log::info!("creating logical device");
        let device =
            create_logical_device(&queue_family_indices, &instance.instance(), physical_device)?;

        let ash_device = device.ash_device();

        log::info!("creating graphics queues");
        let graphics_compute_queue = unsafe {
            ash_device.get_device_queue(
                queue_family_indices
                    .graphics_compute_family
                    .context("failed to get graphics family")?,
                0,
            )
        };

        log::info!("creating present queues");
        let present_queue = unsafe {
            ash_device.get_device_queue(
                queue_family_indices
                    .present_family
                    .context("failed to get present family")?,
                0,
            )
        };

        log::info!("creating swapchain");
        let swapchain = common::swapchain::Swapchain::new(
            &instance.instance(),
            &ash_device,
            physical_device,
            &surface,
            &queue_family_indices,
            window_size,
        )?;

        log::info!("creating command pool");
        let command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(queue_family_indices.graphics_compute_family.unwrap())
                .flags(CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            common::command_pool::CommandPool::new(&command_pool_create_info, ash_device.clone())?
        };

        log::info!("creating command buffers");
        let command_buffers = {
            create_command_buffers(
                &ash_device,
                command_pool.vk_command_pool(),
                MAX_FRAMES_IN_FLIGHT as u32,
            )
        }?;
        let sync_objects = SyncObjects::new(ash_device.clone())?;
        let timer = Timer::new();

        log::info!("creating output render target");
        let output_render_target = ImageBuffer::new(
            &ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(swapchain.vk_swapchain_format())
                .extent(
                    vk::Extent3D::default()
                        .width(window_size.width)
                        .height(window_size.height)
                        .depth(1),
                )
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .array_layers(1)
                .mip_levels(1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .samples(vk::SampleCountFlags::TYPE_1),
            vk::ImageViewType::TYPE_2D,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::ImageAspectFlags::COLOR,
            physical_device,
            ash_device.clone(),
            instance.instance().clone(),
        )?;

        log::info!("creating uniform buffer");
        let transform_ubo = uniform_buffer::UniformBuffer::<TransformParams>::new(
            TransformParams::default(),
            physical_device,
            ash_device.clone(),
            instance.instance().clone(),
        );

        let camera_ubo = uniform_buffer::UniformBuffer::<common::camera::CameraParams>::new(
            common::camera::CameraParams::default(),
            physical_device,
            ash_device.clone(),
            instance.instance().clone(),
        );

        // setup imgui
        log::info!("setting up imgui");
        let imgui_render_pass = {
            let render_pass_attachments = [
                // output
                vk::AttachmentDescription::default()
                    .format(output_render_target.format())
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
            ];

            let color_attachments = [vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachments)];

            let renderpass_create_info = vk::RenderPassCreateInfo::default()
                .attachments(&render_pass_attachments)
                .subpasses(&subpasses);

            common::render_pass::RenderPass::new(renderpass_create_info, ash_device.clone())?
        };

        let mut output_framebuffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let image_views = [output_render_target.image_view()];
            let output_framebuffer = common::framebuffer::Framebuffer::new(
                imgui_render_pass.vk_render_pass(),
                &image_views,
                window_size.width,
                window_size.height,
                ash_device.clone(),
            );
            output_framebuffers.push(output_framebuffer);
        }

        let hidpi_factor = imgui_platform.hidpi_factor();
        let font_size = (20.0 * hidpi_factor) as f32;
        imgui_context
            .fonts()
            .add_font(&[FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            }]);
        imgui_context.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui_platform.attach_window(imgui_context.io_mut(), window, HiDpiMode::Rounded);

        let imgui_renderer = imgui_rs_vulkan_renderer::Renderer::with_default_allocator(
            &instance.instance(),
            physical_device,
            ash_device.clone(),
            graphics_compute_queue,
            command_pool.vk_command_pool(),
            imgui_render_pass.vk_render_pass(),
            imgui_context,
            None,
        )?;

        let camera = common::camera::Camera::new(window_size.width, window_size.height);

        let draw_strategy = draw::deferred::Deferred::new(
            transform_ubo.clone(),
            camera_ubo.clone(),
            scene.clone(),
            &swapchain,
            physical_device,
            ash_device.clone(),
            &surface,
            instance.instance().clone(),
            entry,
        )?;

        Ok(Self {
            device,
            graphics_compute_queue,
            present_queue,
            swapchain,
            command_buffers,
            sync_objects,
            timer,
            output_render_target,
            output_framebuffers,
            imgui_renderer,
            camera,
            transform_ubo,
            camera_ubo,
            imgui_render_pass,
            draw_strategy,
            _command_pool: command_pool,
            _instance: instance,
            _debug_utils: debug_utils,
            _surface: surface,
            _physical_device: physical_device,
        })
    }

    pub fn draw_frame(
        &mut self,
        window: &Window,
        imgui_platform: &mut WinitPlatform,
        imgui_context: &mut imgui::Context,
        control: &control::Control,
    ) {
        let device = self.device.ash_device();

        let fps = {
            let elapsed = self.timer.get_elapsed_and_reset();
            let fps = crate::utils::time::get_fps(&elapsed);
            fps
        };

        log::info!("get next image");
        let (image_index, _is_sub_optimal) = unsafe {
            self.swapchain
                .swapchain_loader()
                .acquire_next_image(
                    self.swapchain.vk_swapchain(),
                    std::u64::MAX,
                    self.sync_objects.image_available_semaphores()[0],
                    vk::Fence::null(),
                )
                .unwrap()
        };
        let command_buffer = self.command_buffers[image_index as usize];

        log::info!("updating camera");
        self.camera.input_control(
            control,
            PhysicalPosition::new(
                self.swapchain.vk_swapchain_extent().width as f32 / 2.0,
                self.swapchain.vk_swapchain_extent().height as f32 / 2.0,
            ),
            self.timer.get_elapsed(),
        );

        log::info!("update uniform buffer");
        let transform_params = self.camera.create_transform_params();
        self.transform_ubo
            .update(image_index as usize, transform_params);
        let camera_params = self.camera.create_camera_params();
        self.camera_ubo.update(image_index as usize, camera_params);

        log::info!("reset command buffer");
        unsafe {
            device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::default())
                .unwrap();
        }

        log::info!("begin command buffers");
        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                .unwrap();
        }

        // draw pass
        self.draw_strategy
            .draw(command_buffer, image_index)
            .unwrap();

        // copy output to output render target
        log::info!("transition image layout");
        let barriers = [
            // Transition draw render target
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.draw_strategy.output_render_target().image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            // Transition output render target
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.output_render_target.image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        // copy output to swapchain
        log::info!("copy draw output to output render target");
        let image_copy = [vk::ImageCopy2::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .extent(Extent3D {
                width: self.swapchain.vk_swapchain_extent().width,
                height: self.swapchain.vk_swapchain_extent().height,
                depth: 1,
            })];
        let copy_image_info = vk::CopyImageInfo2::default()
            .src_image(self.draw_strategy.output_render_target().image())
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(self.output_render_target.image())
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&image_copy);

        unsafe {
            device.cmd_copy_image2(command_buffer, &copy_image_info);
        }

        log::info!("transition image layout");
        let barriers = [
            // Transition output render target
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.output_render_target.image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            // Transition draw render target
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.draw_strategy.output_render_target().image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        // imgui pass
        log::info!("imgui pass");
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.imgui_render_pass.vk_render_pass())
            .framebuffer(
                self.output_framebuffers[image_index as usize]
                    .vk_framebuffer()
                    .clone(),
            )
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.output_render_target.extent().width,
                    height: self.output_render_target.extent().height,
                },
            });

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
        }

        imgui_platform
            .prepare_frame(imgui_context.io_mut(), window)
            .unwrap();
        imgui_context
            .io_mut()
            .update_delta_time(self.timer.get_elapsed());
        let ui = imgui_context.frame();

        ui.window("Stats").build(|| {
            ui.text(format!("FPS: {:.2}", fps));
            let position = control.mouse().position();
            ui.text(format!(
                "mouse position: ({:.2}, {:.2})",
                position.x, position.y
            ));
            ui.text(format!(
                "camera position: ({:.2}, {:.2}, {:.2})",
                self.camera.position().x,
                self.camera.position().y,
                self.camera.position().z
            ));
        });

        ui.window("Control").build(|| {
            let mut mouse_sens = self.camera.mouse_sens();
            ui.slider("mouse sens", 0.0, 1.0, &mut mouse_sens);
            self.camera.set_mouse_sens(mouse_sens);

            let mut move_speed = self.camera.move_speed();
            ui.slider("move speed", 0.0, 1.0, &mut move_speed);
            self.camera.set_move_speed(move_speed);
        });

        imgui_platform.prepare_render(&ui, window);
        self.imgui_renderer
            .cmd_draw(command_buffer, imgui_context.render())
            .unwrap();

        // end render pass
        log::info!("end render pass");
        unsafe {
            device.cmd_end_render_pass(command_buffer);
        }

        log::info!("output: color attachment -> transfer src");
        log::info!("swapchain: present src -> color attachment");
        let barriers = [
            // output
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .image(self.output_render_target.image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            // swapchain
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(self.swapchain.vk_swapchain_images()[image_index as usize])
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        // copy output to swapchain
        log::info!("copy output to swapchain");
        let image_copy = [vk::ImageCopy2::default()
            .src_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .dst_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .extent(Extent3D {
                width: self.swapchain.vk_swapchain_extent().width,
                height: self.swapchain.vk_swapchain_extent().height,
                depth: 1,
            })];
        let copy_image_info = vk::CopyImageInfo2::default()
            .src_image(self.output_render_target.image())
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(self.swapchain.vk_swapchain_images()[image_index as usize])
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&image_copy);

        unsafe {
            device.cmd_copy_image2(command_buffer, &copy_image_info);
        }

        log::info!("output: transfer read -> color attachment");
        log::info!("swapchain: transfer write -> present src");
        let barriers = [
            // output
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.output_render_target.image())
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
            // swapchain
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.swapchain.vk_swapchain_images()[image_index as usize])
                .subresource_range(ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }),
        ];
        let dependency_info = vk::DependencyInfoKHR::default().image_memory_barriers(&barriers);
        unsafe {
            device.cmd_pipeline_barrier2(command_buffer, &dependency_info);
        }

        log::info!("end command buffer");
        unsafe {
            device.end_command_buffer(command_buffer).unwrap();
        }
        log::info!("submit command buffer");
        let command_buffer_infos =
            [CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let signal_semaphore = self.sync_objects.render_finished_semaphores()[image_index as usize];
        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(signal_semaphore)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let wait_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.sync_objects.image_available_semaphores()[0])
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let submit_infos = [vk::SubmitInfo2::default()
            .command_buffer_infos(&command_buffer_infos)
            .signal_semaphore_infos(&signal_semaphore_infos)
            .wait_semaphore_infos(&wait_semaphore_infos)];
        unsafe {
            device
                .queue_submit2(
                    self.graphics_compute_queue,
                    &submit_infos,
                    self.sync_objects.inflight_fences()[image_index as usize],
                )
                .unwrap();
        }

        let swapchains = [self.swapchain.vk_swapchain()];
        let image_indices = [image_index];
        let signal_semaphores = [signal_semaphore];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        log::info!("presenting...");
        unsafe {
            self.swapchain
                .swapchain_loader()
                .queue_present(self.present_queue, &present_info)
                .unwrap()
        };

        // wait
        log::info!("wait for fences");
        let wait_fences = vec![self.sync_objects.inflight_fences()[image_index as usize]];
        unsafe {
            device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .unwrap();
        }

        log::info!("reset fences");
        unsafe {
            device.reset_fences(&wait_fences).unwrap();
        }
    }

    pub fn device_wait_idle(&mut self) {
        unsafe { self.device.ash_device().device_wait_idle().unwrap() };
    }
}
