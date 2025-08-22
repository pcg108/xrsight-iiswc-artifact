#pragma once

#include "../data_format.hpp"
#include "../phonebook.hpp"
#include "third_party/VkBootstrap.h"
#include "vulkan_utils.hpp"

#include <cstdint>

using namespace ILLIXR;

/**
 * @brief Headless sink is an alternative to display sink to run ILLIXR headlessly
 *
 * @details
 * A display sink is a service created with the necessary Vulkan resources to display the rendered images to the screen.
 * It is created either by display_vk, a plugin that configures the Vulkan resources and swapchain,
 * or by monado_vulkan_integration, which populate the Vulkan resources and swapchain from Monado.
 * Previously with the GL implementation, this was not required since we were using GL and Monado was using Vulkan.
 */
class headless_sink : public phonebook::service {
public:
    ~headless_sink() override = default;

    // required by timewarp_vk as a service

    VkInstance       vk_instance;
    VkPhysicalDevice vk_physical_device;
    VkDevice         vk_device;
    VkQueue          graphics_queue;
    uint32_t         graphics_queue_family;


    VkImage                 image;
    VkDeviceMemory          image_memory;
    VkImageView             image_view;
    VkFormat                image_format;
    VkExtent2D              extent;

    // optional
    VmaAllocator vma_allocator;
};
