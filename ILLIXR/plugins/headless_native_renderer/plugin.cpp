#include <array>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <thread>
#include <vector>
#include <ctime>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <vulkan/vulkan_core.h>

#include <filesystem> 

#include <opencv2/opencv.hpp>  
#include <opencv2/imgcodecs.hpp>  
#include <opencv2/core.hpp> 
#include <opencv2/core/mat.hpp>

#define VMA_IMPLEMENTATION
#include "illixr/global_module_defs.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/pose_prediction.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/threadloop.hpp"
// #include "illixr/eye_tracking_host.hpp"
#include "illixr/vk_util/headless_sink.hpp"
#include "illixr/vk_util/render_pass.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "illixr/gl_util/lib/tiny_obj_loader.h"

using namespace ILLIXR;

const record_header mtp_record{"mtp_record",
    {
        {"time_taken", typeid(std::size_t)},
    }};

const char* SOCKET_PATH = "/illixr-host";
const int BUFFER_SIZE = 1024;
const int MAX_CLIENTS = 10;

static constexpr const int width_ = 240;
static constexpr const int height_ = 160;

class native_renderer : public threadloop {
public:
    native_renderer(const std::string& name_, phonebook* pb)
        : threadloop{name_, pb}
        , sb{pb->lookup_impl<switchboard>()}
        , hs{pb->lookup_impl<headless_sink>()}
        , tw{pb->lookup_impl<timewarp>()}
        // , et{pb->lookup_impl<eye_tracking_host>()}
        , src{pb->lookup_impl<app>()}
        , _m_clock{pb->lookup_impl<RelativeClock>()}
        , last_fps_update{std::chrono::duration<long, std::nano>{0}}
        , mtp_logger{record_logger_} {
        spdlogger(std::getenv("NATIVE_RENDERER_LOG_LEVEL"));
    }

    /**
     * @brief Sets up the thread for the plugin.
     *
     * This function initializes depth images, offscreen targets, command buffers, sync objects,
     * application and timewarp passes, offscreen and swapchain framebuffers. Then, it initializes
     * application and timewarp with their respective passes.
     */
    void _p_thread_setup() override {
        for (auto i = 0; i < 2; i++) {
            create_depth_image(&depth_images[i], &depth_image_allocations[i], &depth_image_views[i]);
        }
        for (auto i = 0; i < 2; i++) {
            create_offscreen_target(&offscreen_images[i], &offscreen_image_allocations[i], &offscreen_image_views[i],
                                    &offscreen_framebuffers[i]);
        }
        command_pool            = vulkan_utils::create_command_pool(hs->vk_device, hs->graphics_queue_family);
        app_command_buffer      = vulkan_utils::create_command_buffer(hs->vk_device, command_pool);
        timewarp_command_buffer = vulkan_utils::create_command_buffer(hs->vk_device, command_pool);
        create_sync_objects();
        create_query_pool();
        create_shading_rate_attachment();
        create_app_pass();
        create_timewarp_pass();
        create_sync_objects();
        create_offscreen_framebuffers();
        create_framebuffer();
        src->setup(app_pass, 0);
        tw->setup(timewarp_pass, 0, {std::vector{offscreen_image_views[0]}, std::vector{offscreen_image_views[1]}}, true);


        // open a socket server for the bridge driver
        struct sockaddr_un addr;

        // Create and configure server socket
        if ((server_fd = socket(AF_UNIX, SOCK_SEQPACKET, 0)) < 0) {
            std::cout << "[ILLIXR host server] Error creating server" << std::endl;
            perror("socket");
        }

        
        const char* home = getenv("HOME");
        std::string socket_path = std::string(home) + std::string(SOCKET_PATH);

        // Remove existing socket file
        unlink(socket_path.c_str());

        // Bind socket
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            std::cout << "[ILLIXR host server] Error binding server" << std::endl;
            perror("bind");
            close(server_fd);
        }

        // Listen for connections
        if (listen(server_fd, MAX_CLIENTS) < 0) {
            std::cout << "[ILLIXR host server] Error listening for connections" << std::endl;
            perror("listen");
            close(server_fd);
        }

        // Add server socket to poll list
        struct pollfd server_pollfd;
        server_pollfd.fd = server_fd;
        server_pollfd.events = POLLIN;
        fds.push_back(server_pollfd);

        std::cout << "[ILLIXR host server] ILLIXR server listening on: " << socket_path << std::endl;

        xdma_h2cfd = open("/dev/xdma0_h2c_0", O_WRONLY);
        xdma_c2hfd = open("/dev/xdma0_c2h_0", O_RDONLY);

        if (xdma_h2cfd < 0 || xdma_c2hfd < 0) {
            std::cout << "[ILLIXR host server] Error opening XDMA" << std::endl;
        } else {
            std::cout << "[ILLIXR host server] Opened XDMA" << std::endl;
        }

    }

    /**
     * @brief Executes one iteration of the plugin's main loop.
     *
     * This function handles window events, acquires the next image from the swapchain, updates uniforms,
     * records command buffers, submits commands to the graphics queue, and presents the rendered image.
     * It also handles swapchain recreation if necessary and updates the frames per second (FPS) counter.
     *
     * @throws runtime_error If any Vulkan operation fails.
     */
    void _p_one_iteration() override {

        // Wait for events with 1 second timeout
        int num_ready = poll(fds.data(), fds.size(), 100);
        
        if (num_ready < 0) {
            perror("poll");
            return;
        }

        if (num_ready != 0) {

            // Check all file descriptors
            for (size_t i = 0; i < fds.size(); ++i) {

                if (fds[i].revents == 0)
                    continue;

                // Handle server socket (new connections)
                if (fds[i].fd == server_fd) {
                    if (fds[i].revents & POLLIN) {
                        // Accept new connection
                        int client_fd = accept(server_fd, NULL, NULL);
                        if (client_fd < 0) {
                            perror("accept");
                            continue;
                        }

                        // Check if we have room for more clients
                        if (fds.size() > MAX_CLIENTS + 1) { // +1 for server socket
                            // std::cout << "[guest2bridge] Max clients reached. Rejecting connection." << std::endl;
                            close(client_fd);
                        } else {
                            // Add new client to poll list
                            struct pollfd client_pollfd;
                            client_pollfd.fd = client_fd;
                            client_pollfd.events = POLLIN;
                            fds.push_back(client_pollfd);
                            std::cout << "[ILLIXR host server] New client connected (" << client_fd << ")" << std::endl;
                        }
                    }
                }
                // Handle client socket (data)
                else {
                    if (fds[i].revents & (POLLIN | POLLHUP)) {
                        // Read data from guest
                        ssize_t bytes_read = recv(fds[i].fd, socket_buffer, BUFFER_SIZE, 0);

                        std::cout << bytes_read << std::endl;
                        
                        if (bytes_read <= 0) {
                            // Connection closed or error
                            close(fds[i].fd);
                            fds.erase(fds.begin() + i);
                            --i;
                        } else {

                            uint32_t socket_data[15];
                            std::memcpy(socket_data, socket_buffer, bytes_read);

                            // uint32_t* float_data = reinterpret_cast<uint32_t*>(socket_data);
                            float* float_data = reinterpret_cast<float*>(socket_data);

                            std::cout << "[ILLIXR host server] Received from bridge: ";
                            for (size_t i = 0; i < 12; ++i) {
                                std::cout << float_data[i] << " ";
                            }
                            std::cout << std::endl;

                            int queue_id = float_to_decimal(float_data[0]);
                            int dma_read = float_to_decimal(float_data[1]);
                            // int queue_id = float_data[0];
                            // int dma_read = float_data[1];
                            std::cout << "[ILLIXR host server] Received queue ID: " << queue_id << std::endl;

                            double time_taken = 0;   
                            double bytes_written = 0;                         

                            VK_ASSERT_SUCCESS(vkResetFences(hs->vk_device, 1, &frame_fence))

                            uint32_t response[10];

                            if (queue_id == 0) {

                                auto t = time_point();
                                Eigen::Vector3f v(float_data[2], float_data[3], float_data[4]);
                                Eigen::Quaternionf q(float_data[5], float_data[6], float_data[7], float_data[8]);
                                eye_position_type eye_pos(_m_clock->now(), float_data[9], float_data[10]);

                                int shading_rate = float_to_decimal(float_data[11]);
                                std::cout << "shading rate: " << shading_rate << std::endl;

                                pose_type latest_pose = pose_type(t, v, q);

                                // if we are rendering, use this pose and save it for the timewarp
                                render_pose = latest_pose;

                                // Get the current fast pose and update the uniforms
                                src->update_uniforms(render_pose, render_pose);

                                update_shading_rate(eye_pos.eye_x, eye_pos.eye_y, shading_rate);
                                
                                // Record the command buffer
                                VK_ASSERT_SUCCESS(vkResetCommandBuffer(app_command_buffer, 0))
                                record_app_command_buffer();

                                // Submit the command buffer to the graphics queue

                                VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
                                VkSubmitInfo         application_submit_info{
                                    VK_STRUCTURE_TYPE_SUBMIT_INFO, // sType
                                    nullptr,                        // pNext
                                    0,                             // waitSemaphoreCount
                                    nullptr,                    // pWaitSemaphores
                                    nullptr,                   // pWaitDstStageMask
                                    1,                             // commandBufferCount
                                    &app_command_buffer,           // pCommandBuffers
                                    0,                             // signalSemaphoreCount
                                    &app_render_finished_semaphore // pSignalSemaphores
                                };

                                VK_ASSERT_SUCCESS(vkQueueSubmit(hs->graphics_queue, 1, &application_submit_info, frame_fence))
                                VK_ASSERT_SUCCESS(vkWaitForFences(hs->vk_device, 1, &frame_fence, VK_TRUE, UINT64_MAX))

                                time_taken = get_timestamp(appQueryPool);
                                std::cout << "[ILLIXR host server] render time: " << time_taken / 1e6 << std::endl;

                                response[0] = 1; // responding with 1 packet
                                response[1] = time_taken;

                            } else if (queue_id == 1) {

                                auto t = time_point();
                                Eigen::Vector3f v(float_data[2], float_data[3], float_data[4]);
                                Eigen::Quaternionf q(float_data[5], float_data[6], float_data[7], float_data[8]);
                                eye_position_type eye_pos(_m_clock->now(), float_data[9], float_data[10]);
                                pose_type latest_pose = pose_type(t, v, q);

                                // timewarp will use the pose from render along with the latest pose
                                tw->update_uniforms(render_pose, latest_pose);

                                // Record the command buffer
                                VK_ASSERT_SUCCESS(vkResetCommandBuffer(timewarp_command_buffer, 0))
                                record_tw_command_buffer();

                                VkSubmitInfo timewarp_submit_info{
                                    VK_STRUCTURE_TYPE_SUBMIT_INFO,      // sType
                                    nullptr,                            // pNext
                                    0,                                  // waitSemaphoreCount
                                    nullptr,                            // pWaitSemaphores
                                    nullptr,                            // pWaitDstStageMask
                                    1,                                  // commandBufferCount
                                    &timewarp_command_buffer,           // pCommandBuffers
                                    0,                                  // signalSemaphoreCount
                                    nullptr                              // pSignalSemaphores
                                };

                                VK_ASSERT_SUCCESS(vkQueueSubmit(hs->graphics_queue, 1, &timewarp_submit_info, frame_fence))
                                VK_ASSERT_SUCCESS(vkWaitForFences(hs->vk_device, 1, &frame_fence, VK_TRUE, UINT64_MAX))

                                time_taken = get_timestamp(twQueryPool);
                                std::cout << "[ILLIXR host server] tw time: " << time_taken / 1e6 << std::endl;

                                bytes_written = save_frame(float_data);

                                response[0] = 2; // responding with 2 packets
                                response[1] = time_taken;
                                response[2] = bytes_written;

                            } else if (queue_id == 2) {

                                if (dma_read == 0) {
                                    std::cout << "[ILLIXR host server] Error: received eye tracking request but no XMDA read" << std::endl;
                                }

                                dma_read = 153600;
                                int rc = pread(xdma_c2hfd, input_image_.data(), dma_read, target_dram_addr);
                                if (rc != dma_read) {
                                    std::cout << "[ILLIXR host server] Error: read bytes " << rc << " expected " << dma_read << std::endl;
                                }

                                cv::Mat img = cv::Mat(height_, width_, CV_32FC1, input_image_.data());

                                auto start = _m_clock->now();
                                eye_position_type eye_pos = eye_position_type{_m_clock->now(), 0, 0}; // = et->get_eye_position(img);
                                auto end = _m_clock->now();

                                time_taken = duration2double<std::nano>(end - start);
                                std::cout << "[ILLIXR host server] eye tracking time: " << time_taken << " ns" << std::endl;

                                response[0] = 3; // responding with 3 packets
                                response[1] = time_taken;
                                response[2] = eye_pos.eye_x;
                                response[3] = eye_pos.eye_y;

                            } else {
                                std::cout << "Unrecognized queue ID" << std::endl;
                            }

                            int response_size = sizeof(uint32_t) * (response[0]+1);
                            uint8_t buffer[response_size];
                            std::memcpy(buffer, response, response_size);

                            if (send(fds[i].fd, buffer, response_size, 0) < 0) {
                                std::cout << "[ILLIXR host server] Error responding to bridge driver" << std::endl;
                            }

                            spdlog::get("native_renderer")->info(formatted("time: %d", time_taken));

                            mtp_logger.log(record{mtp_record,
                                {
                                    {(size_t) time_taken},
                                }});
                            

                            VK_ASSERT_SUCCESS(vkResetFences(hs->vk_device, 1, &frame_fence))

                        }
                    }
                }
            }

        }
        

    }

private:

    uint32_t float_to_decimal(const float float_data) {
        uint32_t int_value;
        std::memcpy(&int_value, &float_data, sizeof(uint32_t));
        return int_value;
    }

    double get_timestamp(VkQueryPool qp) {
        uint64_t timestamps[2] = {};
        vkGetQueryPoolResults(hs->vk_device, qp, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);
        return (timestamps[1] - timestamps[0]) * timestampPeriod;
    }

    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        // use image memory barrier (usually used to synchronize access to resources) to transition image layouts and queue family ownership with exclusive sharing mode
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        
        // specify image and specific part of image (not an array and no mipmapping levels so only 1 level and layer specified)
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = 0;
        
        VkPipelineStageFlags sourceStage, destinationStage;
        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR;
            
            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        
        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        endSingleTimeCommands(commandBuffer, VK_NULL_HANDLE);
    }

    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = command_pool;
        allocInfo.commandBufferCount = 1;
        
        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(hs->vk_device, &allocInfo, &commandBuffer);
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        
        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer, VkFence fence) {
        vkEndCommandBuffer(commandBuffer);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        
        vkQueueSubmit(hs->graphics_queue, 1, &submitInfo, fence);
        vkQueueWaitIdle(hs->graphics_queue);
        vkFreeCommandBuffers(hs->vk_device, command_pool, 1, &commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        // has memoryTypes and memoryHeaps
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(hs->vk_physical_device, &memProperties);
        
        // check which memory type has the properties we want
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }

    /**
     * @brief Creates framebuffer
     *
     * @throws runtime_error If framebuffer creation fails.
     */
    void create_framebuffer() {

        std::array<VkImageView, 2> attachments = {hs->image_view, shading_rate_image_views[2]};

        assert(timewarp_pass != VK_NULL_HANDLE);
        VkFramebufferCreateInfo framebuffer_info{
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, // sType
            nullptr,                                   // pNext
            0,                                         // flags
            timewarp_pass,                             // renderPass
            attachments.size(),                        // attachmentCount
            attachments.data(),                        // pAttachments
            hs->extent.width,                           // width
            hs->extent.height,                          // height
            1                                          // layers
        };

        VK_ASSERT_SUCCESS(vkCreateFramebuffer(hs->vk_device, &framebuffer_info, nullptr, &framebuffer))
        
    }

    void record_app_command_buffer() {

        // Begin recording app command buffer
        VkCommandBufferBeginInfo begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, // sType
            nullptr,                                     // pNext
            0,                                           // flags
            nullptr                                      // pInheritanceInfo
        };
        VK_ASSERT_SUCCESS(vkBeginCommandBuffer(app_command_buffer, &begin_info))


        vkCmdResetQueryPool(app_command_buffer, appQueryPool, 0, 2); 
        vkCmdWriteTimestamp(app_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, appQueryPool, 0);

        for (auto eye = 0; eye < 2; eye++) {
            assert(app_pass != VK_NULL_HANDLE);
            std::array<VkClearValue, 2> clear_values = {};
            clear_values[0].color                    = {{1.0f, 1.0f, 1.0f, 1.0f}};
            clear_values[1].depthStencil             = {1.0f, 0};

            
            VkRenderPassBeginInfo render_pass_info{
                VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, // sType
                nullptr,                                  // pNext
                app_pass,                                 // renderPass
                offscreen_framebuffers[eye],              // framebuffer
                {
                    {0, 0},              // offset
                    hs->extent           // extent
                },                       // renderArea
                clear_values.size(),     // clearValueCount
                clear_values.data()      // pClearValues
            };

            vkCmdBeginRenderPass(app_command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

            VkExtent2D                         fragment_size = {1, 1};
            VkFragmentShadingRateCombinerOpKHR combiner_ops[2];
            combiner_ops[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
            combiner_ops[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;

            PFN_vkCmdSetFragmentShadingRateKHR pfnVkCmdSetFragmentShadingRateKHR = 
                (PFN_vkCmdSetFragmentShadingRateKHR)vkGetDeviceProcAddr(hs->vk_device, "vkCmdSetFragmentShadingRateKHR");

            if (pfnVkCmdSetFragmentShadingRateKHR) {
                pfnVkCmdSetFragmentShadingRateKHR(app_command_buffer, &fragment_size, combiner_ops);
            } else {
                throw std::runtime_error("failed to find vkCmdSetFragmentShadingRateKHR");
            }

            // Call app service to record the command buffer
            src->record_command_buffer(app_command_buffer, eye);

            vkCmdEndRenderPass(app_command_buffer);
        }

        vkCmdWriteTimestamp(app_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, appQueryPool, 1);

        VK_ASSERT_SUCCESS(vkEndCommandBuffer(app_command_buffer))

    }

    void record_tw_command_buffer() {

        // Begin recording timewarp command buffer
        VkCommandBufferBeginInfo timewarp_begin_info = {
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, // sType
            nullptr,                                     // pNext
            0,                                           // flags
            nullptr                                      // pInheritanceInfo
        };
        VK_ASSERT_SUCCESS(vkBeginCommandBuffer(timewarp_command_buffer, &timewarp_begin_info)) {

            vkCmdResetQueryPool(timewarp_command_buffer, twQueryPool, 0, 2); 
            vkCmdWriteTimestamp(timewarp_command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, twQueryPool, 0);

            assert(timewarp_pass != VK_NULL_HANDLE);
            VkClearValue          clear_value{.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
            VkRenderPassBeginInfo render_pass_info{
                VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,      // sType
                nullptr,                                       // pNext
                timewarp_pass,                                 // renderPass
                framebuffer,                                    // framebuffer
                {
                    {0, 0},              // offset
                    hs->extent          // extent
                },                       // renderArea
                1,                       // clearValueCount
                &clear_value             // pClearValues
            };

            vkCmdBeginRenderPass(timewarp_command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

            VkExtent2D                         fragment_size = {1, 1};
            VkFragmentShadingRateCombinerOpKHR combiner_ops[2];
            combiner_ops[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
            combiner_ops[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;

            PFN_vkCmdSetFragmentShadingRateKHR pfnVkCmdSetFragmentShadingRateKHR = 
                (PFN_vkCmdSetFragmentShadingRateKHR)vkGetDeviceProcAddr(hs->vk_device, "vkCmdSetFragmentShadingRateKHR");

            if (pfnVkCmdSetFragmentShadingRateKHR) {
                pfnVkCmdSetFragmentShadingRateKHR(timewarp_command_buffer, &fragment_size, combiner_ops);
            } else {
                throw std::runtime_error("failed to find vkCmdSetFragmentShadingRateKHR");
            }


            for (auto eye = 0; eye < 2; eye++) {
                VkViewport viewport{
                    static_cast<float>(hs->extent.width / 2. * eye),            // x
                    0.0f,                                                      // y
                    static_cast<float>(hs->extent.width),                       // width
                    static_cast<float>(hs->extent.height),                      // height
                    0.0f,                                                      // minDepth
                    1.0f                                                       // maxDepth
                };
                vkCmdSetViewport(timewarp_command_buffer, 0, 1, &viewport);

                VkRect2D scissor{
                    {0, 0},              // offset
                    hs->extent          // extent
                };
                vkCmdSetScissor(timewarp_command_buffer, 0, 1, &scissor);

                // Call timewarp service to record the command buffer
                tw->record_command_buffer(timewarp_command_buffer, 0, eye == 0);
            }

            vkCmdEndRenderPass(timewarp_command_buffer);

            vkCmdWriteTimestamp(timewarp_command_buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, twQueryPool, 1);
        }
        VK_ASSERT_SUCCESS(vkEndCommandBuffer(timewarp_command_buffer))

    }

    /**
     * @brief Records the command buffers for a single frame.
     * 
     */

    void record_command_buffer() {


        
    }

    std::string format_float_array_as_path(const float* data, size_t count) {
        namespace fs = std::filesystem;

        // Get current working directory
        fs::path base = fs::current_path() / "saved_frames";

        // Create the directory if it doesn't exist
        if (!fs::exists(base)) {
            fs::create_directory(base);
        }

        // Build filename
        std::ostringstream oss;
        for (size_t i = 0; i < count; ++i) {
            oss << std::fixed << std::setprecision(2) << data[i];
            if (i + 1 < count) {
                oss << "_";
            }
        }
        oss << ".ppm";

        // Return full path
        return (base / oss.str()).string();
    }

    int save_frame(float* float_data) {

        // create image in host memory 
        VkImage dstImage;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = hs->extent.width;
        imageInfo.extent.height = hs->extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = hs->image_format;
        imageInfo.tiling = VK_IMAGE_TILING_LINEAR;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage(hs->vk_device, &imageInfo, nullptr, &dstImage) != VK_SUCCESS) {
            throw std::runtime_error("failed to create destination image!");
        }
        
        VkDeviceMemory dstImageMemory;

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(hs->vk_device, dstImage, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        
        if (vkAllocateMemory(hs->vk_device, &allocInfo, nullptr, &dstImageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }
        vkBindImageMemory(hs->vk_device, dstImage, dstImageMemory, 0);

        // transition dstImage to optimal layout for recieving the image
        transitionImageLayout(dstImage, hs->image_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        // copy image
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageCopy imageCopyRegion{};
        imageCopyRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.srcSubresource.layerCount = 1;
        imageCopyRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        imageCopyRegion.dstSubresource.layerCount = 1;
        imageCopyRegion.extent.width = hs->extent.width;
        imageCopyRegion.extent.height = hs->extent.height;
        imageCopyRegion.extent.depth = 1;
        
        vkCmdCopyImage(
                       commandBuffer,
                       hs->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       dstImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &imageCopyRegion);
        
        // submit command buffer
        endSingleTimeCommands(commandBuffer, copy_frame_fence);

        // wait for copy to complete
        vkWaitForFences(hs->vk_device, 1, &copy_frame_fence, VK_TRUE, UINT64_MAX);
        
        // transition image to general layout to write to file later
        transitionImageLayout(dstImage, VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

        // Get layout of the image (including row pitch)
        VkImageSubresource subResource{};
        subResource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        VkSubresourceLayout subResourceLayout;
        vkGetImageSubresourceLayout(hs->vk_device, dstImage, &subResource, &subResourceLayout);
        
        // Map image memory to a pointer so we can start copying from it
        const char* imagedata;
        vkMapMemory(hs->vk_device, dstImageMemory, 0, VK_WHOLE_SIZE, 0, (void**)&imagedata);
        imagedata += subResourceLayout.offset;

        // write the image data to XDMA
        int rc = pwrite(xdma_h2cfd, imagedata, hs->extent.height * hs->extent.width, target_dram_addr);
        if (rc < 0) {
            std::cout << "[ILLIXR host server] error writing " << rc << " bytes to XDMA" << std::endl;
        } else {
            std::cout << "[ILLIXR host server] wrote " << rc << " bytes to XDMA" << std::endl;
        }

        
        // std::string fname = formatted("/scratch/prashanth/ILLIXR/build/saved_frames/%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f.ppm", 
        //                                                                             float_data[2],
        //                                                                             float_data[3],
        //                                                                             float_data[4],
        //                                                                             float_data[5],
        //                                                                             float_data[6],
        //                                                                             float_data[7],
        //                                                                             float_data[8],
        //                                                                             float_data[9],
        //                                                                             float_data[10]);

        std::string fname = format_float_array_as_path(float_data, 9);

        const char* filename = fname.c_str();

        std::ofstream file(filename, std::ofstream::binary);
        // ppm header
        file << "P6\n" << hs->extent.width << "\n" << hs->extent.height << "\n" << 255 << "\n";

        
        for (int32_t y = 0; y < hs->extent.height; y++) {
            unsigned int *row = (unsigned int*)imagedata;
            for (int32_t x = 0; x < hs->extent.width; x++) {
                
                // swizzle colors because format is VK_FORMAT_B8G8R8A8_SRGB, so switch BGR to RGB
                file.write((char*)row+2, 1);
                file.write((char*)row+1, 1);
                file.write((char*)row, 1);
                row++;
            }
            imagedata += subResourceLayout.rowPitch;
        }
        file.close();


        std::cout << "[ILLIXR host server] saved " << fname.c_str() << std::endl;
        
        // reset fence for copy operation
        vkResetFences(hs->vk_device, 1, &copy_frame_fence);
        
        // unmap and free memory
        vkUnmapMemory(hs->vk_device, dstImageMemory);
        vkFreeMemory(hs->vk_device, dstImageMemory, nullptr);
        vkDestroyImage(hs->vk_device, dstImage, nullptr);
        

        frame_count += 1;

        return rc;
    }

    void create_query_pool() {

        VkQueryPoolCreateInfo queryPoolCreateInfo = {};
        queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryPoolCreateInfo.queryCount = 2;

        VK_ASSERT_SUCCESS(vkCreateQueryPool(hs->vk_device, &queryPoolCreateInfo, nullptr, &appQueryPool)) 
        VK_ASSERT_SUCCESS(vkCreateQueryPool(hs->vk_device, &queryPoolCreateInfo, nullptr, &twQueryPool)) 

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(hs->vk_physical_device, &deviceProperties);

        // The timestampPeriod is given in nanoseconds per timestamp unit.
        timestampPeriod = deviceProperties.limits.timestampPeriod;
        std::cout << "Timestamp period: " << timestampPeriod << " ns" << std::endl;
    }

    /**
     * @brief Creates synchronization objects for the application.
     *
     * This function creates a timeline semaphore for the application render finished signal,
     * a binary semaphore for the image available signal, a binary semaphore for the timewarp render finished signal,
     * and a fence for frame synchronization.
     *
     * @throws runtime_error If any Vulkan operation fails.
     */
    void create_sync_objects() {
        VkSemaphoreTypeCreateInfo timeline_semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, // sType
            nullptr,                                      // pNext
            VK_SEMAPHORE_TYPE_TIMELINE,                   // semaphoreType
            0                                             // initialValue
        };

        VkSemaphoreCreateInfo create_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, // sType
            &timeline_semaphore_info,                // pNext
            0                                        // flags
        };

        vkCreateSemaphore(hs->vk_device, &create_info, nullptr, &app_render_finished_semaphore);

        VkSemaphoreCreateInfo semaphore_info{
            VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, // sType
            nullptr,                                 // pNext
            0                                        // flags
        };
        VkFenceCreateInfo fence_info{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
            nullptr,                             // pNext
            VK_FENCE_CREATE_SIGNALED_BIT         // flags
        };

        VK_ASSERT_SUCCESS(vkCreateSemaphore(hs->vk_device, &semaphore_info, nullptr, &image_available_semaphore))
        VK_ASSERT_SUCCESS(vkCreateSemaphore(hs->vk_device, &semaphore_info, nullptr, &timewarp_render_finished_semaphore))
        VK_ASSERT_SUCCESS(vkCreateFence(hs->vk_device, &fence_info, nullptr, &frame_fence))

        VkFenceCreateInfo copy_fence_info{
            VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
            nullptr,                             // pNext
        };
        VK_ASSERT_SUCCESS(vkCreateFence(hs->vk_device, &copy_fence_info, nullptr, &copy_frame_fence))
    }

    /**
     * @brief Creates a depth image for the application.
     * @param depth_image Pointer to the depth image handle.
     * @param depth_image_allocation Pointer to the depth image memory allocation handle.
     * @param depth_image_view Pointer to the depth image view handle.
     */
    void create_depth_image(VkImage* depth_image, VmaAllocation* depth_image_allocation, VkImageView* depth_image_view) {
        VkImageCreateInfo image_info{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
            nullptr,                             // pNext
            0,                                   // flags
            VK_IMAGE_TYPE_2D,                    // imageType
            VK_FORMAT_D32_SFLOAT,                // format
            {
                display_params::width_pixels,                                         // width
                display_params::height_pixels,                                        // height
                1                                                                     // depth
            },                                                                        // extent
            1,                                                                        // mipLevels
            1,                                                                        // arrayLayers
            VK_SAMPLE_COUNT_1_BIT,                                                    // samples
            VK_IMAGE_TILING_OPTIMAL,                                                  // tiling
            VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // usage
            {},                                                                       // sharingMode
            0,                                                                        // queueFamilyIndexCount
            nullptr,                                                                  // pQueueFamilyIndices
            {}                                                                        // initialLayout
        };

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

        VK_ASSERT_SUCCESS(
            vmaCreateImage(hs->vma_allocator, &image_info, &alloc_info, depth_image, depth_image_allocation, nullptr))

        VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
            nullptr,                                  // pNext
            0,                                        // flags
            *depth_image,                             // image
            VK_IMAGE_VIEW_TYPE_2D,                    // viewType
            VK_FORMAT_D32_SFLOAT,                     // format
            {},                                       // components
            {
                VK_IMAGE_ASPECT_DEPTH_BIT, // aspectMask
                0,                         // baseMipLevel
                1,                         // levelCount
                0,                         // baseArrayLayer
                1                          // layerCount
            }                              // subresourceRange
        };

        VK_ASSERT_SUCCESS(vkCreateImageView(hs->vk_device, &view_info, nullptr, depth_image_view))
    }

    /**
     * @brief Creates an offscreen target for the application to render to.
     * @param offscreen_image Pointer to the offscreen image handle.
     * @param offscreen_image_allocation Pointer to the offscreen image memory allocation handle.
     * @param offscreen_image_view Pointer to the offscreen image view handle.
     * @param offscreen_framebuffer Pointer to the offscreen framebuffer handle.
     */
    void create_offscreen_target(VkImage* offscreen_image, VmaAllocation* offscreen_image_allocation,
                                 VkImageView* offscreen_image_view, [[maybe_unused]] VkFramebuffer* offscreen_framebuffer) {
        VkImageCreateInfo image_info{
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
            nullptr,                             // pNext
            0,                                   // flags
            VK_IMAGE_TYPE_2D,                    // imageType
            VK_FORMAT_B8G8R8A8_UNORM,            // format
            {
                display_params::width_pixels,                                 // width
                display_params::height_pixels,                                // height
                1                                                             // depth
            },                                                                // extent
            1,                                                                // mipLevels
            1,                                                                // arrayLayers
            VK_SAMPLE_COUNT_1_BIT,                                            // samples
            VK_IMAGE_TILING_OPTIMAL,                                          // tiling
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, // usage
            {},                                                               // sharingMode
            0,                                                                // queueFamilyIndexCount
            nullptr,                                                          // pQueueFamilyIndices
            {}                                                                // initialLayout
        };

        VmaAllocationCreateInfo alloc_info{.usage = VMA_MEMORY_USAGE_GPU_ONLY};

        VK_ASSERT_SUCCESS(
            vmaCreateImage(hs->vma_allocator, &image_info, &alloc_info, offscreen_image, offscreen_image_allocation, nullptr))

        VkImageViewCreateInfo view_info{
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
            nullptr,                                  // pNext
            0,                                        // flags
            *offscreen_image,                         // image
            VK_IMAGE_VIEW_TYPE_2D,                    // viewType
            VK_FORMAT_B8G8R8A8_UNORM,                 // format
            {},                                       // components
            {
                VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
                0,                         // baseMipLevel
                1,                         // levelCount
                0,                         // baseArrayLayer
                1                          // layerCount
            }                              // subresourceRange
        };

        VK_ASSERT_SUCCESS(vkCreateImageView(hs->vk_device, &view_info, nullptr, offscreen_image_view))
    }

    /**
     * @brief Creates the offscreen framebuffers for the application.
     */
    void create_offscreen_framebuffers() {
        for (auto eye = 0; eye < 2; eye++) {
            std::array<VkImageView, 3> attachments = {offscreen_image_views[eye], depth_image_views[eye], shading_rate_image_views[eye]};

            assert(app_pass != VK_NULL_HANDLE);
            VkFramebufferCreateInfo framebuffer_info{
                VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, // sType
                nullptr,                                   // pNext
                0,                                         // flags
                app_pass,                                  // renderPass
                static_cast<uint32_t>(attachments.size()), // attachmentCount
                attachments.data(),                        // pAttachments
                display_params::width_pixels,              // width
                display_params::height_pixels,             // height
                1                                          // layers
            };

            VK_ASSERT_SUCCESS(vkCreateFramebuffer(hs->vk_device, &framebuffer_info, nullptr, &offscreen_framebuffers[eye]))
        }
    }

    void update_shading_rate(float eye_x, float eye_y, int shading_rate) {

        // Populate the buffer with lowest possible shading rate pattern (4x4)

        uint8_t  val; 
        if (shading_rate == 0) {
            val = (1 >> 1) | (1 << 1);
        } else if (shading_rate == 1) {
            val = (2 >> 1) | (2 << 1);
        } else if (shading_rate == 2) {
            val = (4 >> 1) | (4 << 1);
        } else {
            std::cout << "[ILLIXR headless_native_renderer] Invalid shading rate: " << shading_rate << std::endl;
            throw std::runtime_error("Invalid shading rate");
        }

                          
        // std::cout << "Initial: " << static_cast<int>(val) << std::endl;
        uint8_t *shading_rate_pattern_data = new uint8_t[buffer_size];
        memset(shading_rate_pattern_data, val, buffer_size);

        // update the pattern with the fovea position
        uint8_t *ptrData = shading_rate_pattern_data;
        for (uint32_t y = 0; y < image_extent.height; y++)
        {
            for (uint32_t x = 0; x < image_extent.width; x++)
            {
                const float deltaX = ((eye_x*0.6) - static_cast<float>(x)) / image_extent.width * 100.0f;
                const float deltaY = ((eye_y*0.6) - static_cast<float>(y)) / image_extent.height * 100.0f;
                const float dist   = std::sqrt(deltaX * deltaX + deltaY * deltaY);

                if (dist < 10) {
                    *ptrData = 0;
                }

                ptrData++;
            }
        }

        
        for (auto eye = 0; eye < 3; eye++) {

            auto* shading_buffer_ptr = shading_rate_alloc_infos[eye].pMappedData;
            std::memcpy(shading_buffer_ptr, shading_rate_pattern_data, buffer_size);
            
            // for (int y = 20; y < 70; y++) {
            //     for (int x = 60; x < 100; x++) {
            //         std::cout << std::setw(2) << static_cast<int>(shading_rate_pattern_data[y*image_extent.width + x]) << " ";
            //     }
            //     std::cout << std::endl;
            // }

            // Upload the buffer containing the shading rates to the image that'll be used as the shading rate attachment inside our renderpass
            transitionImageLayout(shading_rate_images[eye], VK_FORMAT_R8_UINT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkCommandBuffer copy_cmd = beginSingleTimeCommands();

            VkBufferImageCopy buffer_copy_region           = {};
            buffer_copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            buffer_copy_region.imageSubresource.layerCount = 1;
            buffer_copy_region.imageExtent.width           = image_extent.width;
            buffer_copy_region.imageExtent.height          = image_extent.height;
            buffer_copy_region.imageExtent.depth           = 1;
            vkCmdCopyBufferToImage(copy_cmd, shading_rate_buffers[eye], shading_rate_images[eye], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &buffer_copy_region);

            // Transfer image layout to fragment shading rate attachment layout required to access this in the renderpass
            endSingleTimeCommands(copy_cmd, nullptr);

            transitionImageLayout(shading_rate_images[eye], VK_FORMAT_R8_UINT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR);
        
        }

    }

    void create_shading_rate_attachment() {

        get_shading_rate_properties();

        // Check if the requested format for the shading rate attachment supports the required flag
	    VkFormat           requested_format = VK_FORMAT_R8_UINT;
	    VkFormatProperties format_properties;
	    vkGetPhysicalDeviceFormatProperties(hs->vk_physical_device, requested_format, &format_properties);
        if (!(format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR))
        {
            throw std::runtime_error("Selected shading rate attachment image format does not support required formate featureflag");
        }

        // Shading rate image size depends on shading rate texel size
        // For each texel in the target image, there is a corresponding shading texel size width x height block in the shading rate image
        image_extent.width  = static_cast<uint32_t>(ceil(display_params::width_pixels  / static_cast<float>(shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.width)));
        image_extent.height = static_cast<uint32_t>(ceil(display_params::height_pixels / static_cast<float>(shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.height)));
        image_extent.depth  = 1;

        std::cout << "width: " << static_cast<uint32_t>(image_extent.width) << " height: " << static_cast<uint32_t>(image_extent.height) << std::endl;

        // Allocate a buffer that stores the shading rates
        buffer_size = image_extent.width * image_extent.height * sizeof(uint8_t);

        // Sstaging buffers for shading rate pattern
        create_shading_rate_buffers(buffer_size);

        for (auto eye = 0; eye < 3; eye++) {


            VkImageCreateInfo image_create_info{};
            image_create_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            image_create_info.imageType     = VK_IMAGE_TYPE_2D;
            image_create_info.format        = VK_FORMAT_R8_UINT;
            image_create_info.extent        = image_extent;
            image_create_info.mipLevels     = 1;
            image_create_info.arrayLayers   = 1;
            image_create_info.samples       = VK_SAMPLE_COUNT_1_BIT;
            image_create_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
            image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_create_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
            image_create_info.usage         = VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            VK_ASSERT_SUCCESS(vkCreateImage(hs->vk_device, &image_create_info, nullptr, &shading_rate_images[eye]))
            VkMemoryRequirements memory_requirements{};
            vkGetImageMemoryRequirements(hs->vk_device, shading_rate_images[eye], &memory_requirements);

            VkMemoryAllocateInfo memory_allocate_info{};
            memory_allocate_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            memory_allocate_info.allocationSize  = memory_requirements.size;
            memory_allocate_info.memoryTypeIndex = findMemoryType(memory_requirements.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VK_ASSERT_SUCCESS(vkAllocateMemory(hs->vk_device, &memory_allocate_info, nullptr, &shading_rate_memories[eye]))
            VK_ASSERT_SUCCESS(vkBindImageMemory(hs->vk_device, shading_rate_images[eye], shading_rate_memories[eye], 0))

            VkImageViewCreateInfo image_view_create_info{};
            image_view_create_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            image_view_create_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
            image_view_create_info.image                           = shading_rate_images[eye];
            image_view_create_info.format                          = VK_FORMAT_R8_UINT;
            image_view_create_info.subresourceRange.baseMipLevel   = 0;
            image_view_create_info.subresourceRange.levelCount     = 1;
            image_view_create_info.subresourceRange.baseArrayLayer = 0;
            image_view_create_info.subresourceRange.layerCount     = 1;
            image_view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
            VK_ASSERT_SUCCESS(vkCreateImageView(hs->vk_device, &image_view_create_info, nullptr, &shading_rate_image_views[eye]))
        }
    }


    void create_shading_rate_buffers(size_t buffer_size) {

        for (auto eye = 0; eye < 3; eye++) {
            VkBufferCreateInfo bufferInfo = {
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
                nullptr,                              // pNext
                0,                                    // flags
                0,                                    // size
                0,                                    // usage
                {},                                   // sharingMode
                0,                                    // queueFamilyIndexCount
                nullptr                               // pQueueFamilyIndices
            };
            bufferInfo.size  = buffer_size;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    
            VmaAllocationCreateInfo createInfo = {};
            createInfo.usage                   = VMA_MEMORY_USAGE_AUTO;
            createInfo.flags         = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            createInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            VK_ASSERT_SUCCESS(vmaCreateBuffer(hs->vma_allocator, &bufferInfo, &createInfo, &shading_rate_buffers[eye], &shading_rate_allocs[eye], &shading_rate_alloc_infos[eye]))
        }
        

        
    }

    void get_shading_rate_properties() {
        // Get fragment shading rate properties
        shadingRateProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;

        VkPhysicalDeviceProperties2 deviceProps2 = {};
        deviceProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        deviceProps2.pNext = &shadingRateProperties;
        vkGetPhysicalDeviceProperties2(hs->vk_physical_device, &deviceProps2);

        // Get fragment shading rate features
        VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures = {};
        shadingRateFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;

        VkPhysicalDeviceFeatures2 deviceFeatures2 = {};
        deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        deviceFeatures2.pNext = &shadingRateFeatures;
        vkGetPhysicalDeviceFeatures2(hs->vk_physical_device, &deviceFeatures2);

        // Check if pipeline and attachment shading rates are supported
        if (!shadingRateFeatures.pipelineFragmentShadingRate || 
            !shadingRateFeatures.attachmentFragmentShadingRate) {
            printf("Required fragment shading rate features not supported\n");
        }

        // Store the max fragment size and combiner modes for later use
        uint32_t maxFragmentSize[2] = {
            shadingRateProperties.maxFragmentSize.width,
            shadingRateProperties.maxFragmentSize.height
        };

        printf("Maximum fragment shading rate size: %dx%d\n", 
            maxFragmentSize[0], maxFragmentSize[1]);
    }




    /**
     * @brief Creates a render pass for the application.
     *
     * This function sets up the attachment descriptions for color and depth, the attachment references,
     * the subpass description, and the subpass dependencies. It then creates a render pass with these configurations.
     *
     * @throws runtime_error If render pass creation fails.
     */
    void create_app_pass() {

        std::array<VkAttachmentDescription2KHR, 3> attachmentDescriptions = {};
        // Color attachment
        attachmentDescriptions[0].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachmentDescriptions[0].format         = VK_FORMAT_B8G8R8A8_UNORM;
        attachmentDescriptions[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescriptions[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentDescriptions[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // Depth attachment
        attachmentDescriptions[1].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachmentDescriptions[1].format         = VK_FORMAT_D32_SFLOAT;
        attachmentDescriptions[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentDescriptions[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        // Fragment shading rate attachment
        attachmentDescriptions[2].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachmentDescriptions[2].format         = VK_FORMAT_R8_UINT;
        attachmentDescriptions[2].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[2].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachmentDescriptions[2].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[2].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[2].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[2].initialLayout  = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
        attachmentDescriptions[2].finalLayout    = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;


        VkAttachmentReference2 color_attachment_ref = {};
        color_attachment_ref.sType                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        color_attachment_ref.attachment                = 0;
        color_attachment_ref.layout                    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment_ref.aspectMask                = VK_IMAGE_ASPECT_COLOR_BIT;
    
        VkAttachmentReference2 depth_attachment_ref = {};
        depth_attachment_ref.sType                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        depth_attachment_ref.attachment                = 1;
        depth_attachment_ref.layout                    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment_ref.aspectMask                = VK_IMAGE_ASPECT_DEPTH_BIT;
    
        // Setup the attachment reference for the shading rate image attachment in slot 2
        VkAttachmentReference2 fragment_shading_rate_reference = {};
        fragment_shading_rate_reference.sType                  = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        fragment_shading_rate_reference.attachment             = 2;
        fragment_shading_rate_reference.layout                 = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
    
        // Setup the attachment info for the shading rate image, which will be added to the sub pass via structure chaining (in pNext)
        VkFragmentShadingRateAttachmentInfoKHR fragment_shading_rate_attachment_info = {};
        fragment_shading_rate_attachment_info.sType                                  = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
        fragment_shading_rate_attachment_info.pFragmentShadingRateAttachment         = &fragment_shading_rate_reference;
        fragment_shading_rate_attachment_info.shadingRateAttachmentTexelSize.width   = shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.width;
        fragment_shading_rate_attachment_info.shadingRateAttachmentTexelSize.height  = shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.height;

        VkSubpassDescription2 subpass = {};
        subpass.sType                    = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        subpass.pipelineBindPoint        = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount     = 1;
        subpass.pColorAttachments        = &color_attachment_ref;
        subpass.pDepthStencilAttachment  = &depth_attachment_ref;
        subpass.inputAttachmentCount     = 0;
        subpass.pInputAttachments        = nullptr;
        subpass.preserveAttachmentCount  = 0;
        subpass.pPreserveAttachments     = nullptr;
        subpass.pResolveAttachments      = nullptr;
        subpass.pNext                    = &fragment_shading_rate_attachment_info;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency2, 2> dependencies = {};

        dependencies[0].sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass      = 0;
        dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        dependencies[1].sType           = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
        dependencies[1].srcSubpass      = 0;
        dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        VkRenderPassCreateInfo2 render_pass_info = {};
            render_pass_info.sType                      = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
            render_pass_info.attachmentCount            = static_cast<uint32_t>(attachmentDescriptions.size());
            render_pass_info.pAttachments               = attachmentDescriptions.data();
            render_pass_info.subpassCount               = 1;
            render_pass_info.pSubpasses                 = &subpass;
            render_pass_info.dependencyCount            = static_cast<uint32_t>(dependencies.size());
            render_pass_info.pDependencies              = dependencies.data();

        VK_ASSERT_SUCCESS(vkCreateRenderPass2(hs->vk_device, &render_pass_info, nullptr, &app_pass));
    }

    /**
     * @brief Creates a render pass for timewarp.
     */
    void create_timewarp_pass() {

        std::array<VkAttachmentDescription2KHR, 2> attachmentDescriptions = {};
        attachmentDescriptions[0].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachmentDescriptions[0].format         = hs->image_format;
        attachmentDescriptions[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescriptions[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescriptions[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        attachmentDescriptions[0].finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        // Fragment shading rate attachment
        attachmentDescriptions[1].sType          = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachmentDescriptions[1].format         = VK_FORMAT_R8_UINT;
        attachmentDescriptions[1].samples        = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescriptions[1].loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachmentDescriptions[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescriptions[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescriptions[1].initialLayout  = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
        attachmentDescriptions[1].finalLayout    = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;

        VkAttachmentReference2 color_attachment_ref = {};
        color_attachment_ref.sType                     = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        color_attachment_ref.attachment                = 0;
        color_attachment_ref.layout                    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        color_attachment_ref.aspectMask                = VK_IMAGE_ASPECT_COLOR_BIT;

        // Setup the attachment reference for the shading rate image attachment in slot 2
        VkAttachmentReference2 fragment_shading_rate_reference = {};
        fragment_shading_rate_reference.sType                  = VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2;
        fragment_shading_rate_reference.attachment             = 1;
        fragment_shading_rate_reference.layout                 = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;

         // Setup the attachment info for the shading rate image, which will be added to the sub pass via structure chaining (in pNext)
         VkFragmentShadingRateAttachmentInfoKHR fragment_shading_rate_attachment_info = {};
         fragment_shading_rate_attachment_info.sType                                  = VK_STRUCTURE_TYPE_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR;
         fragment_shading_rate_attachment_info.pFragmentShadingRateAttachment         = &fragment_shading_rate_reference;
         fragment_shading_rate_attachment_info.shadingRateAttachmentTexelSize.width   = shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.width;
         fragment_shading_rate_attachment_info.shadingRateAttachmentTexelSize.height  = shadingRateProperties.maxFragmentShadingRateAttachmentTexelSize.height;

        VkSubpassDescription2 subpass = {};
        subpass.sType                    = VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2;
        subpass.pipelineBindPoint        = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount     = 1;
        subpass.pColorAttachments        = &color_attachment_ref;
        subpass.pDepthStencilAttachment  = nullptr;
        subpass.inputAttachmentCount     = 0;
        subpass.pInputAttachments        = nullptr;
        subpass.preserveAttachmentCount  = 0;
        subpass.pPreserveAttachments     = nullptr;
        subpass.pResolveAttachments      = nullptr;
        subpass.pNext                    = &fragment_shading_rate_attachment_info;

        VkRenderPassCreateInfo2 render_pass_info = {};
            render_pass_info.sType                      = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2;
            render_pass_info.attachmentCount            = static_cast<uint32_t>(attachmentDescriptions.size());
            render_pass_info.pAttachments               = attachmentDescriptions.data();
            render_pass_info.subpassCount               = 1;
            render_pass_info.pSubpasses                 = &subpass;
            render_pass_info.dependencyCount            = 0;
            render_pass_info.pDependencies              = nullptr;

        VK_ASSERT_SUCCESS(vkCreateRenderPass2(hs->vk_device, &render_pass_info, nullptr, &timewarp_pass));
    }

    template <typename... Args>
    std::string formatted(const char* format, Args&&... args) {
        int size = std::snprintf(nullptr, 0, format, std::forward<Args>(args)...) + 1;
        if (size <= 0) {
            throw std::runtime_error("Error during formatting.");
        }        
        std::string result(size, '\0');        
        std::snprintf(&result[0], size, format, std::forward<Args>(args)...);        
        result.resize(size - 1); // remove the null terminator
        return result;
    }

    const std::shared_ptr<switchboard>         sb;
    // const std::shared_ptr<eye_tracking_host> et;
    const std::shared_ptr<headless_sink>       hs;
    const std::shared_ptr<timewarp>            tw;
    const std::shared_ptr<app>                 src;
    const std::shared_ptr<const RelativeClock> _m_clock;

    VkCommandPool   command_pool{};
    VkCommandBuffer app_command_buffer{};
    VkCommandBuffer timewarp_command_buffer{};

    std::array<VkImage, 2>       depth_images{};
    std::array<VmaAllocation, 2> depth_image_allocations{};
    std::array<VkImageView, 2>   depth_image_views{};

    std::array<VkImage, 2>       offscreen_images{};
    std::array<VmaAllocation, 2> offscreen_image_allocations{};
    std::array<VkImageView, 2>   offscreen_image_views{};
    std::array<VkFramebuffer, 2> offscreen_framebuffers{};

    VkFramebuffer framebuffer;

    VkRenderPass app_pass{};
    VkRenderPass timewarp_pass{};

    VkQueryPool appQueryPool;
    VkQueryPool twQueryPool;

    VkSemaphore image_available_semaphore{};
    VkSemaphore app_render_finished_semaphore{};
    VkSemaphore timewarp_render_finished_semaphore{};
    VkFence     frame_fence{};
    VkFence     copy_frame_fence{};

    VkPhysicalDeviceFragmentShadingRatePropertiesKHR shadingRateProperties = {};
    VkExtent3D image_extent{};
    VkDeviceSize buffer_size;

    std::array<VkImage, 3> shading_rate_images;
    std::array<VkDeviceMemory, 3> shading_rate_memories;
    std::array<VkImageView, 3> shading_rate_image_views;

    std::array<VkBuffer, 3> shading_rate_buffers{};
    std::array<VmaAllocation, 3> shading_rate_allocs{};
    std::array<VmaAllocationInfo, 3> shading_rate_alloc_infos{};

    uint64_t timeline_semaphore_value = 1;

    int        fps{};
    time_point last_fps_update;

    int frame_count = 0;
    record_coalescer mtp_logger;

    int server_fd;
    std::vector<struct pollfd> fds;
    uint8_t socket_buffer[BUFFER_SIZE];

    float timestampPeriod;
    pose_type render_pose;

    int xdma_h2cfd;
    int xdma_c2hfd;
    unsigned long long target_dram_addr = (0x88000000 + 0x380000000) % 0x400000000;
    std::array<float, width_ * height_> input_image_{}; 

    bool once = true;
};
PLUGIN_MAIN(native_renderer)
