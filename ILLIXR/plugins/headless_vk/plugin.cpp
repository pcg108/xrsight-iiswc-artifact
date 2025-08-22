#define VMA_IMPLEMENTATION
#include "illixr/data_format.hpp"
#include "illixr/phonebook.hpp"
#include "illixr/switchboard.hpp"
#include "illixr/threadloop.hpp"
#include "illixr/vk_util/headless_sink.hpp"

using namespace ILLIXR;

class headless_vk : public headless_sink {
public:
    explicit headless_vk(const phonebook* const pb)
        : sb{pb->lookup_impl<switchboard>()} { }

    /**
     * @brief This function sets up the Vulkan environments. See headless_sink::setup().
     */
    void setup() {
        setup_vk();
    }


private:



    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        // has memoryTypes and memoryHeaps
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(vk_physical_device, &memProperties);
        
        // check which memory type has the properties we want
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type!");
    }


    /**
     * @brief Sets up the Vulkan environment.
     *
     * This function initializes the Vulkan instance, selects the physical device, creates the Vulkan device,
     * gets the graphics queues, and sets up the VMA allocator.
     *
     * @throws runtime_error If any of the Vulkan setup steps fail.
     */
    void setup_vk() {
        vkb::InstanceBuilder builder;
        auto                 instance_ret =
            builder.set_headless(true).set_app_name("ILLIXR Vulkan Headless")
                .require_api_version(1, 2)
                .request_validation_layers()
                .enable_validation_layers()
                .set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                       VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                       const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) -> VkBool32 {
                    auto severity = vkb::to_string_message_severity(messageSeverity);
                    auto type     = vkb::to_string_message_type(messageType);
                    spdlog::get("illixr")->debug("[headless_vk] [{}: {}] {}", severity, type, pCallbackData->pMessage);
                    return VK_FALSE;
                })
                .build();
        if (!instance_ret) {
            ILLIXR::abort("Failed to create Vulkan instance. Error: " + instance_ret.error().message());
        }
        vkb_instance = instance_ret.value();
        vk_instance  = vkb_instance.instance;

        vkb::PhysicalDeviceSelector selector{vkb_instance};

        VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR,
            .pNext = nullptr,
            .pipelineFragmentShadingRate = VK_TRUE,
            .attachmentFragmentShadingRate = VK_TRUE
        };

        auto physical_device_ret = selector.set_minimum_version(1, 2)
                                       .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
                                       .add_required_extension(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)
                                       .add_required_extension_features(shadingRateFeatures)
                                       .select();

        if (!physical_device_ret) {
            ILLIXR::abort("Failed to select Vulkan Physical Device. Error: " + physical_device_ret.error().message());
        } 
        physical_device    = physical_device_ret.value();
        vk_physical_device = physical_device.physical_device;

        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(vk_physical_device, &deviceProperties);

        std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;

        vkb::DeviceBuilder device_builder{physical_device};

        // enable timeline semaphore
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_semaphore_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
            nullptr, // pNext
            VK_TRUE  // timelineSemaphore
        };

        // enable anisotropic filtering
        auto device_ret = device_builder.add_pNext(&timeline_semaphore_features).build();
        if (!device_ret) {
            ILLIXR::abort("Failed to create Vulkan device. Error: " + device_ret.error().message());
        }
        vkb_device = device_ret.value();
        vk_device  = vkb_device.device;

        auto graphics_queue_ret = vkb_device.get_queue(vkb::QueueType::graphics);
        if (!graphics_queue_ret) {
            ILLIXR::abort("Failed to get Vulkan graphics queue. Error: " + graphics_queue_ret.error().message());
        }
        graphics_queue        = graphics_queue_ret.value();
        graphics_queue_family = vkb_device.get_queue_index(vkb::QueueType::graphics).value();

        ////// create image ///////
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = display_params::width_pixels;
        imageInfo.extent.height = display_params::height_pixels;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = VK_FORMAT_B8G8R8A8_SRGB;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;

        if (vkCreateImage(vk_device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        /////// create image memory ///////

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(vk_device, image, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        
        if (vkAllocateMemory(vk_device, &allocInfo, nullptr, &image_memory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }
        vkBindImageMemory(vk_device, image, image_memory, 0);

        /////// create image view ///////

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_B8G8R8A8_SRGB;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        
        if (vkCreateImageView(vk_device, &viewInfo, nullptr, &image_view) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        image_format    =   VK_FORMAT_B8G8R8A8_SRGB;
        extent          =   VkExtent2D{display_params::width_pixels, display_params::height_pixels};

        vma_allocator = vulkan_utils::create_vma_allocator(vk_instance, vk_physical_device, vk_device);
    }

    const std::shared_ptr<switchboard> sb;
    vkb::Instance                      vkb_instance;
    vkb::PhysicalDevice                physical_device;
    vkb::Device                        vkb_device;

    std::atomic<bool> should_poll{true};

    friend class headless_vk_plugin;
};

class headless_vk_plugin : public plugin {
public:
    headless_vk_plugin(const std::string& name, phonebook* pb)
        : plugin{name, pb}
        , _hvk{std::make_shared<headless_vk>(pb)}
        , _pb{pb} {
        _pb->register_impl<headless_sink>(std::static_pointer_cast<headless_sink>(_hvk));
    }

    void start() override {
        main_thread = std::thread(&headless_vk_plugin::main_loop, this);
        while (!ready) {
            // yield
            std::this_thread::yield();
        }
    }

    void stop() override {
        running = false;
    }

private:
    std::thread                 main_thread;
    std::atomic<bool>           ready{false};
    std::shared_ptr<headless_vk> _hvk;
    std::atomic<bool>           running{true};
    phonebook*                  _pb;

    void main_loop() {
        _hvk->setup();

        ready = true;

        while (running) {
            
        }
    }
};

PLUGIN_MAIN(headless_vk_plugin)