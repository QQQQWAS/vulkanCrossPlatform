#include <cstddef>
#include <cstdint>
// #include <limits>
#include <vector>
#include <cstdio>
#include <readFile/readFile.hpp>
#include <platformBridge/glue.hpp>
#include <platformBridge/baseState.hpp>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <vulkan/vulkan_core.h>

const int FRAMES_IN_FLIGHT = 1;

#define err(...) {fprintf(stderr, __VA_ARGS__); exit(0);}

struct ShaderSource{
  std::string name;
  std::string source;
  VkShaderStageFlagBits stage;
};

#ifdef ANDROID
#include <time.h>


// from android samples
/* return current time in milliseconds */
static double glfwGetTime(void) {
    struct timespec res;
    clock_gettime(CLOCK_MONOTONIC, &res);
    return (1000.0 * res.tv_sec + (double) res.tv_nsec / 1e6)/1000;
}
#endif

struct Buffer{
  VkDevice _device;
  
  VkBuffer buffer;
  VkMemoryRequirements memRequirements;
  VkDeviceMemory memory;
  
  Buffer(){}
  void create(VkDevice device, VkPhysicalDeviceMemoryProperties physicalDeviceMemProperties, size_t size, VkBufferUsageFlags usageFlags){
    _device = device;
    VkResult result = vkCreateBuffer(_device, Ptr(VkBufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .size = size,
      .usage = usageFlags,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .queueFamilyIndexCount = 0,
      .pQueueFamilyIndices = NULL,
    }), NULL, &buffer);
    if(result != VK_SUCCESS) throw(result);

    vkGetBufferMemoryRequirements(_device, buffer, &memRequirements);
    uint32_t memoryTypeIndex = 0;
    for(; memoryTypeIndex  < physicalDeviceMemProperties.memoryTypeCount; memoryTypeIndex ++){
      if((memRequirements.memoryTypeBits) & (1 << memoryTypeIndex ) && physicalDeviceMemProperties.memoryTypes[memoryTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
        break;
      }
    }
    if(memoryTypeIndex == physicalDeviceMemProperties.memoryTypeCount) throw(VK_ERROR_MEMORY_MAP_FAILED); // idk if this is the correct error code
    vkAllocateMemory(_device, Ptr(VkMemoryAllocateInfo{
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = NULL,
      .allocationSize = memRequirements.size,
      .memoryTypeIndex = memoryTypeIndex,
    }), NULL, &memory);
    vkBindBufferMemory(_device, buffer, memory, 0);
  }
  Buffer(VkDevice device, VkPhysicalDeviceMemoryProperties physicalDeviceMemProperties, size_t size, VkBufferUsageFlags usageFlags){
    create(device, physicalDeviceMemProperties, size, usageFlags);
  }
  void* map(){
    void* mem;
    vkMapMemory(_device, memory, 0, memRequirements.size, 0, &mem);    
    return mem;
  }
  void unmap(){
    vkUnmapMemory(_device, memory);    
  }
  void write(void* data, size_t size){
    void* mem = map();
    memcpy(mem, data, size);
    unmap();
  }
  void destroy(){
    vkFreeMemory(_device, memory, NULL);
    vkDestroyBuffer(_device, buffer, NULL);
    memory = 0;
    buffer = 0;
  }
  ~Buffer(){
    if(buffer && memory) destroy();
  }
  operator VkBuffer&(){
    return buffer;
  }
  operator VkMemoryRequirements&(){
    return memRequirements;
  }
  operator VkDeviceMemory&(){
    return memory;
  }
  operator VkBuffer*(){
    return &buffer;
  }
  operator VkMemoryRequirements*(){
    return &memRequirements;
  }
  operator VkDeviceMemory*(){
    return &memory;
  }
};

struct Surface{
  VkInstance _instance;
  
  VkSurfaceKHR surface;
  VkExtent2D extent;
  VkSurfaceCapabilitiesKHR capabilities;
  VkSurfaceFormatKHR format;
  VkPresentModeKHR presentMode;

  void create(VkInstance instance, VkPhysicalDevice physicalDevice, VkSurfaceKHR in_surface, VkExtent2D in_extent){
    _instance = instance;
    surface = in_surface;
    extent = in_extent;
  
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());
    format = formats[0];

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());
    presentMode = presentModes[0];
    // surfacePresentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    // surfacePresentMode = VK_PRESENT_MODE_FIFO_KHR;
  }
  Surface(VkInstance instance, VkPhysicalDevice physicalDevice, VkSurfaceKHR in_surface, VkExtent2D in_extent){
    create(instance, physicalDevice, in_surface, in_extent);
  }
  void destroy(){
    vkDestroySurfaceKHR(_instance, surface, NULL);
  }
  operator VkSurfaceKHR&(){
    return surface;
  }
  operator VkSurfaceKHR*(){
    return &surface;
  }
};

struct Swapchain{
  VkDevice _device;

  VkSwapchainCreateInfoKHR createInfo;
  VkSwapchainKHR swapchain;
  std::vector<VkImage> images;
  std::vector<VkImageView> imageViews;

  Swapchain(){}
  VkResult populateImageVectors(VkFormat format){
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(_device, swapchain, &imageCount, nullptr);
    images.resize(imageCount);
    imageViews.resize(imageCount);
    vkGetSwapchainImagesKHR(_device, swapchain, &imageCount, images.data());

    for(uint32_t i = 0; i < imageCount; i++){
      VkResult result = vkCreateImageView(_device, Ptr(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .image = images[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components = {
          .r = VK_COMPONENT_SWIZZLE_IDENTITY,
          .g = VK_COMPONENT_SWIZZLE_IDENTITY,
          .b = VK_COMPONENT_SWIZZLE_IDENTITY,
          .a = VK_COMPONENT_SWIZZLE_IDENTITY,          
        },
        .subresourceRange = {
          .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
        },
      }), NULL, &imageViews[i]);
      if(result != VK_SUCCESS) return result;
    }
    return VK_SUCCESS;
  }
  void setup(VkDevice device, Surface& surface, uint32_t graphicsQueueIdx, uint32_t presentQueueIdx){
    _device = device;

    bool sameQueue = graphicsQueueIdx == presentQueueIdx;
    uint32_t queueIndicesArr[] = {graphicsQueueIdx, presentQueueIdx};
    createInfo = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
      .pNext = NULL,
      .flags = 0,
      .surface = surface,
      .minImageCount = surface.capabilities.minImageCount,
      .imageFormat = surface.format.format,
      .imageColorSpace = surface.format.colorSpace,
      .imageExtent = surface.extent,
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode =      sameQueue ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
      .queueFamilyIndexCount = (uint32_t)2,    // graphics and present
      .pQueueFamilyIndices =   queueIndicesArr,
      .preTransform = surface.capabilities.currentTransform,
    #ifdef ANDROID
      .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
    #else
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    #endif
      .presentMode = surface.presentMode,
      .clipped = VK_TRUE,
      .oldSwapchain = VK_NULL_HANDLE,
    };
  }
  VkResult create(Surface& surface){
    VkResult result = vkCreateSwapchainKHR(_device, &createInfo, NULL, &swapchain);    
    if(result != VK_SUCCESS) return result;
    return populateImageVectors(surface.format.format);
  }
  Swapchain(VkDevice device, Surface& surface, uint32_t graphicsQueueIdx, uint32_t presentQueueIdx){
    setup(device, surface, graphicsQueueIdx, presentQueueIdx);
    create(surface);
  }
  void destroy(){
    for(auto& imv: imageViews) vkDestroyImageView(_device, imv, NULL);
    vkDestroySwapchainKHR(_device, swapchain, NULL);
  }
  operator VkSwapchainKHR&(){
    return swapchain;
  }
  operator VkSwapchainKHR*(){
    return &swapchain;
  }
};

void pbMain(BaseState& appState){
  printf("pbMain()\n");
  // Instance
  VkInstance instance;
  if(vkCreateInstance(Ptr(VkInstanceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .pApplicationInfo = Ptr(VkApplicationInfo{
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pNext = NULL,
      .pApplicationName = "PlatformBridge",
      .applicationVersion = VK_MAKE_VERSION(0, 0, 1),
      .pEngineName = "PlatformBridge",
      .engineVersion = VK_MAKE_VERSION(0, 0, 1),
      .apiVersion = VK_API_VERSION_1_3,
    }),
    .enabledLayerCount = 1,
    .ppEnabledLayerNames = std::vector{"VK_LAYER_KHRONOS_validation"}.data(),
    .enabledExtensionCount = (uint32_t)appState.vkInstanceExtensions.size(),
    .ppEnabledExtensionNames = appState.vkInstanceExtensions.data(),
  }), NULL, &instance) != VK_SUCCESS) err("Failed to create instance\n");
  
  // Physical device
  VkPhysicalDevice physicalDevice;
  { //choose physical device
    uint32_t deviceCount;
    vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // physicalDevice = devices[0];
    physicalDevice = devices[devices.size()-1];
  }
  {
    VkPhysicalDeviceProperties physDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physDeviceProperties);
    printf("Using physical device %s\n", physDeviceProperties.deviceName);
  }

  VkPhysicalDeviceMemoryProperties physicalDeviceMemProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemProperties);
  
  // Surface
  Surface surface(instance, physicalDevice, appState.getSurface(instance), appState.getSurfaceExtent());

  // Queues
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  uint32_t graphicsQueueIdx;
  VkQueue presentQueue = VK_NULL_HANDLE;
  uint32_t presentQueueIdx;
  bool sameQueue = false;
  { // choose suitable queues from queue families of physicalDevice
    uint32_t propertyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &propertyCount, NULL);
    std::vector<VkQueueFamilyProperties> properties(propertyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &propertyCount, properties.data());

    bool graphicsQueueFound = 0;
    bool presentQueueFound = 0;
    for(uint32_t i = 0; i < propertyCount; i++){
      auto prop = properties[i];
      if(graphicsQueueFound == 0){
        if(prop.queueFlags & VK_QUEUE_GRAPHICS_BIT){
          graphicsQueueFound = 1;
          graphicsQueueIdx = i;
        }
      }
      if(presentQueueFound == 0){
        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &present);
        if(present){
          presentQueueFound = 1;
          presentQueueIdx = i;
        }
      }
    }
    if(!graphicsQueueFound){
      fprintf(stderr, "Graphics queue not found\n");
      exit(0);
    }
    if(!presentQueueFound){
      fprintf(stderr, "Present queue not found\n");
      exit(0);
    }
    sameQueue = graphicsQueueIdx == presentQueueIdx;
  }

  // Logical device
  VkDevice device;
  std::vector<uint32_t> uniqueQueueIndices;
  uniqueQueueIndices.push_back(graphicsQueueIdx);
  if(!sameQueue) uniqueQueueIndices.push_back(presentQueueIdx);
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  for(uint32_t q: uniqueQueueIndices){
    queueCreateInfos.push_back({
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .queueFamilyIndex = q,
      .queueCount = 1,
      .pQueuePriorities = Ptr(1.0f),
    });
  }
  
  VkResult result = vkCreateDevice(physicalDevice, Ptr(VkDeviceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
    .pQueueCreateInfos = queueCreateInfos.data(),
    .enabledExtensionCount = 1,
    
    .ppEnabledExtensionNames = Ptr("VK_KHR_swapchain"),
    .pEnabledFeatures = {},
  }), NULL, &device);
  if(result != VK_SUCCESS) err("Failed to create device %d\n", result);
  vkGetDeviceQueue(device, graphicsQueueIdx, 0, &graphicsQueue);
  vkGetDeviceQueue(device, presentQueueIdx, 0, &presentQueue);

  // Swapchain
  Swapchain swapchain(device, surface, graphicsQueueIdx, presentQueueIdx);

  // Pipeline :0
  VkPipeline pipeline;

  VkDescriptorSetLayout pipelineDescriptorSetLayout;
  vkCreateDescriptorSetLayout(device, Ptr(VkDescriptorSetLayoutCreateInfo{
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .bindingCount = 2,
    .pBindings = std::vector{
      VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = NULL,
      },
      VkDescriptorSetLayoutBinding{
        .binding = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = NULL,
      },
    }.data(),
  }), NULL, &pipelineDescriptorSetLayout);

  VkPipelineLayout pipelineLayout;
  if(vkCreatePipelineLayout(device, Ptr(VkPipelineLayoutCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .setLayoutCount = 1,
    .pSetLayouts = &pipelineDescriptorSetLayout,
    .pushConstantRangeCount = 0,
    .pPushConstantRanges = NULL,
  }), NULL, &pipelineLayout)) err("Failed to create pipeline layout\n");

  VkRenderPass pipelineRenderPass;
  if(vkCreateRenderPass(device, Ptr(VkRenderPassCreateInfo{
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .attachmentCount = 1,
    .pAttachments = Ptr(VkAttachmentDescription{
      .flags = 0,
      .format = surface.format.format,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    }),
    .subpassCount = 1,
    .pSubpasses = Ptr(VkSubpassDescription{
      .flags = 0,
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .inputAttachmentCount = 0,
      .pInputAttachments = NULL,
      .colorAttachmentCount = 1,
      .pColorAttachments = Ptr(VkAttachmentReference{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      }),
      .pResolveAttachments = 0,
      .pDepthStencilAttachment = NULL,
      .preserveAttachmentCount = 0,
      .pPreserveAttachments = NULL,
    }),
    .dependencyCount = 1,
    .pDependencies = Ptr(VkSubpassDependency{
      .srcSubpass = VK_SUBPASS_EXTERNAL,
      .dstSubpass = 0,
      .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      .srcAccessMask = 0,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      // .dependencyFlags = NULL,
    }),
  }), NULL, &pipelineRenderPass) != VK_SUCCESS) err("Failed to create render pass\n");

  std::vector<ShaderSource> shaderSources = { //first abstractions yay
    {
      .name = "Basic vert",
      .source = appState.readFile("shaders/basic.vert.spv"),
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
    },
    {
      .name = "Basic frag",
      .source = appState.readFile("shaders/basic.frag.spv"),
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
    }
  };
  std::vector<VkPipelineShaderStageCreateInfo> shaderStages(shaderSources.size());
  for(size_t i = 0; i < shaderSources.size(); i++){
    ShaderSource& ss = shaderSources[i];
    if(ss.source == "") err("Shader \"%s\" source empty\n", ss.name.data());
    VkShaderModule module;
    if(vkCreateShaderModule(device, Ptr(VkShaderModuleCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .codeSize = ss.source.size() - 1, //null termination?
      .pCode = (uint32_t*)ss.source.data(),
    }), NULL, &module) != VK_SUCCESS) err("Failed to create shader module for \"%s\"\n", ss.name.data());
    shaderStages[i] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .stage = ss.stage,
      .module = module,
      .pName = "main",
      .pSpecializationInfo = NULL, //VERY COL THING MAYBE      
    };
  }

  struct Vertex{
    glm::vec2 pos;
    glm::vec3 col;
  };

  if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, Ptr(VkGraphicsPipelineCreateInfo{
    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
    .pNext = NULL,
    .flags = 0,
    .stageCount = (uint32_t)shaderStages.size(),
    .pStages = shaderStages.data(),
    .pVertexInputState = Ptr(VkPipelineVertexInputStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = Ptr(VkVertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
      }),
      .vertexAttributeDescriptionCount = 2,
      .pVertexAttributeDescriptions = std::vector{
        VkVertexInputAttributeDescription{
          .location = 0,
          .binding = 0,
          .format = VK_FORMAT_R32G32_SFLOAT,
          .offset = offsetof(Vertex, pos),
        },
        VkVertexInputAttributeDescription{
          .location = 1,
          .binding = 0,
          .format = VK_FORMAT_R32G32B32_SFLOAT,
          .offset = offsetof(Vertex, col),
        },
      }.data(),
    }),
    .pInputAssemblyState = Ptr(VkPipelineInputAssemblyStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
    }),
    .pTessellationState = NULL,
    .pViewportState = Ptr(VkPipelineViewportStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .viewportCount = 1,
      .pViewports = NULL,
      // .pViewports = Ptr(VkViewport{ // static viewport
      //    .x = 0,
      //    .y = 0,
      //    .width =  (float)surface.extent.width,
      //    .height = (float)surface.extent.height,
      //    .minDepth = 0,
      //    .maxDepth = 0,
      //  }),
      .scissorCount = 1,
      .pScissors = NULL,
      // .pScissors = Ptr(VkRect2D{ // static scissor
      //    .offset = {0, 0},
      //    .extent = surface.extent,
      //  }),
    }),
    .pRasterizationState = Ptr(VkPipelineRasterizationStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .depthBiasConstantFactor = 0,
      .depthBiasClamp = 0,
      .depthBiasSlopeFactor = 0,
      .lineWidth = 1,
    }),
    .pMultisampleState = Ptr(VkPipelineMultisampleStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
      .sampleShadingEnable = VK_FALSE,
      .minSampleShading = 1,
      .pSampleMask = NULL,
      .alphaToCoverageEnable = VK_FALSE,
      .alphaToOneEnable = VK_FALSE,
    }),
    .pDepthStencilState = NULL,
    .pColorBlendState = Ptr(VkPipelineColorBlendStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .logicOpEnable = VK_FALSE,
      .logicOp = VK_LOGIC_OP_COPY,
      .attachmentCount = 1,
      .pAttachments = Ptr(VkPipelineColorBlendAttachmentState{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      }),
      .blendConstants = {0, 0, 0, 0},
    }),
    .pDynamicState = Ptr(VkPipelineDynamicStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .dynamicStateCount = 2, //for dynamic viewport and scissor
      .pDynamicStates = std::vector<VkDynamicState>{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
      }.data(),
    }),
    .layout = pipelineLayout,
    .renderPass = pipelineRenderPass,
    .subpass = 0,
    .basePipelineHandle = VK_NULL_HANDLE,
    .basePipelineIndex = 0,
  }), NULL, &pipeline) != VK_SUCCESS) err("Failed to create pipeline\n");

  // Descriptor pool
  VkDescriptorPool descriptorPool;
  if(vkCreateDescriptorPool(device, Ptr(VkDescriptorPoolCreateInfo{
    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
    .pNext = NULL,
    .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
    .maxSets = FRAMES_IN_FLIGHT*2,
    .poolSizeCount = 2,
    .pPoolSizes = std::vector{
      VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = FRAMES_IN_FLIGHT,
      },
      VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = FRAMES_IN_FLIGHT,
      },
    }.data(),
  }), NULL, &descriptorPool) != VK_SUCCESS) err("Failed to create descriptor pool\n");

  // Descriptor sets
  std::vector<VkDescriptorSet> descriptorSets(FRAMES_IN_FLIGHT);
  {
    std::vector<VkDescriptorSetLayout> layouts(FRAMES_IN_FLIGHT, pipelineDescriptorSetLayout);
    if(vkAllocateDescriptorSets(device, Ptr(VkDescriptorSetAllocateInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .pNext = NULL,
      .descriptorPool = descriptorPool,
      .descriptorSetCount = FRAMES_IN_FLIGHT,
      .pSetLayouts = layouts.data(),
    }), descriptorSets.data()) != VK_SUCCESS) err("Failed to allocate descriptor sets\n");
  }
  
  // Vertex buffer
  Vertex vertices[3] = {
    {{0, 0}, {1, 0, 0}},
    {{1, 0}, {0, 1, 0}},
    {{0, 1}, {0, 0, 1}},
  };
  Buffer vertexBuffer(device, physicalDeviceMemProperties, sizeof(vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  vertexBuffer.write(vertices, sizeof(vertices));

  // Uniform buffer
  // TODO
  // Uniform<struct{
  //   glm::mat4 transform;
  //   glm::mat4 project;
  //   glm::vec3 idk;
  // }> uniformVert();
  //
  // Uniform<struct{
  //   glm::vec3 sunPos;
  //   glm::vec3 sunCol;
  // }> uniformVert();
  //
  // please do this thank you <3 <3
  //
  std::vector<Buffer> uniformBuffersVert(FRAMES_IN_FLIGHT);
  std::vector<Buffer> uniformBuffersFrag(FRAMES_IN_FLIGHT);
  std::vector<void*> uniformMapsVert(FRAMES_IN_FLIGHT);
  std::vector<void*> uniformMapsFrag(FRAMES_IN_FLIGHT);
  for(int i = 0; i < FRAMES_IN_FLIGHT; i++){
    uniformBuffersVert[i].create(device, physicalDeviceMemProperties, sizeof(glm::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    uniformBuffersFrag[i].create(device, physicalDeviceMemProperties, sizeof(glm::vec3), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    uniformMapsVert[i] = uniformBuffersVert[i].map();
    uniformMapsFrag[i] = uniformBuffersFrag[i].map();
  }
  
  for(int i = 0; i < FRAMES_IN_FLIGHT; i++){
    vkUpdateDescriptorSets(device, 2, std::vector{
      VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSets[i],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = Ptr(VkDescriptorBufferInfo{
          .buffer = uniformBuffersVert[i],
          .offset = 0,
          .range = sizeof(glm::mat4),
        }),
        .pTexelBufferView = NULL,    
      },
      VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = descriptorSets[i],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pImageInfo = NULL,
        .pBufferInfo = Ptr(VkDescriptorBufferInfo{
          .buffer = uniformBuffersFrag[i],
          .offset = 0,
          .range = sizeof(glm::vec3),
        }),
        .pTexelBufferView = NULL,    
      },
    }.data(), 0, NULL);
  }

  // Framebuffers
  std::vector<VkFramebuffer> swapchainFramebuffers(swapchain.images.size());
  // std::vector<VkFramebuffer> swapchainFramebuffers(FRAMES_IN_FLIGHT);
  for(uint32_t i = 0; i < (uint32_t)swapchainFramebuffers.size(); i++){
    if(vkCreateFramebuffer(device, Ptr(VkFramebufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,
      .renderPass = pipelineRenderPass,
      .attachmentCount = 1,
      .pAttachments = &swapchain.imageViews[i],
      .width = surface.extent.width,
      .height = surface.extent.height,
      .layers = 1,
    }), NULL, &swapchainFramebuffers[i]) != VK_SUCCESS) err("Failed to create framebuffer %d\n", i);
  }

  // Command pool
  VkCommandPool cmdPool;
  if(vkCreateCommandPool(device, Ptr(VkCommandPoolCreateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    .pNext = NULL,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = graphicsQueueIdx,
  }), NULL, &cmdPool) != VK_SUCCESS) err("Failed to create command pool\n");

  // Command buffers
  std::vector<VkCommandBuffer> cmdBuffers(FRAMES_IN_FLIGHT);
  if(vkAllocateCommandBuffers(device, Ptr(VkCommandBufferAllocateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    .pNext = NULL,
    .commandPool = cmdPool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = FRAMES_IN_FLIGHT,
  }), cmdBuffers.data()) != VK_SUCCESS) err("Failed to allocate command buffers\n");

  // Semaphores
  std::vector<VkSemaphore> imageReadySemaphores(FRAMES_IN_FLIGHT);
  for(VkSemaphore& s: imageReadySemaphores){
    if(vkCreateSemaphore(device, Ptr(VkSemaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,      
    }), NULL, &s) != VK_SUCCESS) err("Failed to create semaphore\n");
  }
  std::vector<VkSemaphore> renderDoneSemaphores(swapchain.images.size());
  for(VkSemaphore& s: renderDoneSemaphores){
    if(vkCreateSemaphore(device, Ptr(VkSemaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = NULL,
      .flags = 0,      
    }), NULL, &s) != VK_SUCCESS) err("Failed to create semaphore\n");
  }

  // Fences
  std::vector<VkFence> renderDoneFences(FRAMES_IN_FLIGHT);
  for(VkFence& f: renderDoneFences){
    if(vkCreateFence(device, Ptr(VkFenceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      .pNext = NULL,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    }), NULL, &f) != VK_SUCCESS) err("Failed to create fence\n");
  }

  // Main loop OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO OvO
  uint32_t frameInFlightIdx = 0;
  double time = 0;
  double dt = 0.016;
  double timeCurrFrame, timeLastFrame;
  while(appState.pollEvents()){
    timeLastFrame = timeCurrFrame;
    VkSemaphore imageReadySemaphore = imageReadySemaphores[frameInFlightIdx];
    VkFence renderDoneFence = renderDoneFences[frameInFlightIdx];
    VkCommandBuffer cmdBuffer = cmdBuffers[frameInFlightIdx];

    glm::mat4* transformUniformVert = (glm::mat4*)uniformMapsVert[frameInFlightIdx];
    *transformUniformVert = glm::rotate(glm::mat4(1), (float)time, glm::vec3(0, 0, 1));

    glm::vec3* transformUniformFrag = (glm::vec3*)uniformMapsFrag[frameInFlightIdx];
    *transformUniformFrag = glm::vec3(glm::sin(time)*0.5+0.5, glm::cos(time)*0.5+0.5, 0.5);

    vkWaitForFences(device, 1, &renderDoneFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &renderDoneFence);
    
    if(appState.surfaceSizeChanged){
      // printf("Hello\n");
      surface.extent = appState.getSurfaceExtent();
      if((surface.extent.width == 0 || surface.extent.height == 0) && appState.pollEvents()){}; // if minimized then wait
      appState.surfaceSizeChanged = false;

      vkDeviceWaitIdle(device);
      for(auto i: swapchainFramebuffers){
        vkDestroyFramebuffer(device, i, NULL);
      }
      
      swapchain.createInfo.imageExtent = surface.extent;
      swapchain.destroy();
      swapchain.create(surface);

      for(uint32_t i = 0; i < swapchainFramebuffers.size(); i++){
        if(vkCreateFramebuffer(device, Ptr(VkFramebufferCreateInfo{
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          .pNext = NULL,
          .flags = 0,
          .renderPass = pipelineRenderPass,
          .attachmentCount = 1,
          .pAttachments = &swapchain.imageViews[i],
          .width = surface.extent.width,
          .height = surface.extent.height,
          .layers = 1,
        }), NULL, &swapchainFramebuffers[i]) != VK_SUCCESS) err("Failed to create framebuffer %d\n", i);
      }
    }
    uint32_t swapchainIdx;
    VkResult swapchainImageState = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageReadySemaphore, VK_NULL_HANDLE, &swapchainIdx);
    if(swapchainImageState != VK_SUCCESS) printf("acquire %d\n", swapchainImageState);
    VkFramebuffer framebuffer = swapchainFramebuffers[swapchainIdx];
    VkSemaphore renderDoneSemaphore = renderDoneSemaphores[swapchainIdx];

    vkResetCommandBuffer(cmdBuffer, 0);
    vkBeginCommandBuffer(cmdBuffer, Ptr(VkCommandBufferBeginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .pNext = NULL,
      .flags = 0,
      .pInheritanceInfo = NULL,      
    }));
    vkCmdBeginRenderPass(cmdBuffer, Ptr(VkRenderPassBeginInfo{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .pNext = NULL,
      
      .renderPass = pipelineRenderPass,
      .framebuffer = framebuffer,
      .renderArea = {
        .offset = {0, 0},
        .extent = surface.extent,
      },
      .clearValueCount = 1,
      .pClearValues = Ptr(VkClearValue{
        .color = {.float32 = {0, 0, 0, 0}},
      }),
    }), VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkViewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)(surface.extent.width),
      .height = (float)(surface.extent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor{
      .offset = {0, 0},
      .extent = surface.extent,
    };
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, vertexBuffer, Ptr<VkDeviceSize>(0));
    vkCmdBindDescriptorSets(
      cmdBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipelineLayout,
      0,
      1,
      &descriptorSets[frameInFlightIdx],
      0,
      NULL
    );

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuffer);
    if(vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) err("Failed to end command buffer\n");

    VkResult queueSubmitResult = vkQueueSubmit(graphicsQueue, 1, Ptr(VkSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = NULL,
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &imageReadySemaphore,
      .pWaitDstStageMask = Ptr(VkPipelineStageFlags{VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT}),

      .commandBufferCount = 1,
      .pCommandBuffers = &cmdBuffer,
      
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &renderDoneSemaphore,
    }), renderDoneFence);
    if(queueSubmitResult != VK_SUCCESS) err("Failed to submit queue %d\n", queueSubmitResult);

    // if(vkQueuePresentKHR(presentQueue, Ptr(VkPresentInfoKHR{
    vkQueuePresentKHR(presentQueue, Ptr(VkPresentInfoKHR{
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .pNext = NULL,
      
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &renderDoneSemaphore,
      
      .swapchainCount = 1,
      .pSwapchains = swapchain,
      .pImageIndices = &swapchainIdx,
      .pResults = NULL,
    }));
    // })) != VK_SUCCESS) err("Failed to present queue\n");

    timeCurrFrame = glfwGetTime();
    dt = timeCurrFrame - timeLastFrame;
    time += dt;
    // printf("FPS: %f\n", 1/dt);
    frameInFlightIdx = (frameInFlightIdx + 1)%FRAMES_IN_FLIGHT;
  }

  printf("End\n");
  vkDeviceWaitIdle(device);
  
  for(VkFence f: renderDoneFences){
    vkDestroyFence(device, f, NULL);
  }
  for(VkSemaphore s: imageReadySemaphores){
    vkDestroySemaphore(device, s, NULL);
  }
  for(VkSemaphore s: renderDoneSemaphores){
    vkDestroySemaphore(device, s, NULL);
  }
  vkDestroyCommandPool(device, cmdPool, NULL);
  for(VkFramebuffer fb: swapchainFramebuffers){
    vkDestroyFramebuffer(device, fb, NULL);
  }
  vertexBuffer.destroy();
  for(auto& ub: uniformBuffersVert) ub.destroy();
  for(auto& ub: uniformBuffersFrag) ub.destroy();
  for(VkPipelineShaderStageCreateInfo& ssc: shaderStages){
    vkDestroyShaderModule(device, ssc.module, NULL);
  }
  vkFreeDescriptorSets(device, descriptorPool, FRAMES_IN_FLIGHT, descriptorSets.data());
  vkDestroyDescriptorSetLayout(device, pipelineDescriptorSetLayout, NULL);
  vkDestroyDescriptorPool(device, descriptorPool, NULL);
  vkDestroyPipeline(device, pipeline, NULL);
  vkDestroyRenderPass(device, pipelineRenderPass, NULL);
  vkDestroyPipelineLayout(device, pipelineLayout, NULL);
  swapchain.destroy();
  vkDestroyDevice(device, NULL);
  surface.destroy();
  vkDestroyInstance(instance, NULL);
}
