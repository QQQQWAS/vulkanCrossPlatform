#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <cstdio>
#include <readFile/readFile.hpp>
#include <platformBridge/glue.hpp>
#include <platformBridge/baseState.hpp>

#include <vulkan/vulkan_core.h>

const int FRAMES_IN_FLIGHT = 5;

#define err(...) {fprintf(stderr, __VA_ARGS__); exit(0);}

struct ShaderSource{
  std::string name;
  std::string source;
  VkShaderStageFlagBits stage;
};

void pbMain(BaseState& appState){
  printf("pbMain()\n");

  // Instance
  VkInstance instance;
  if(vkCreateInstance(Ptr(VkInstanceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    // .pNext = NULL,
    // .flags = NULL,
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

    physicalDevice = devices[devices.size()-1];
  }
  VkPhysicalDeviceProperties physDeviceProperties;
  vkGetPhysicalDeviceProperties(physicalDevice, &physDeviceProperties);
  printf("Using physical device %s\n", physDeviceProperties.deviceName);

  // Device memory
  VkPhysicalDeviceMemoryProperties physicalDeviceMemProperties;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemProperties);
  
  // Surface
  VkSurfaceKHR surface = appState.getSurface(instance);
  
  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities);

  VkExtent2D surfaceExtent = surfaceCapabilities.currentExtent;
  if((surfaceExtent.width  == std::numeric_limits<uint32_t>::max()) ||
     (surfaceExtent.height == std::numeric_limits<uint32_t>::max())){
    surfaceExtent = appState.getSurfaceExtent();
  }
  printf("Surface size: %dx%d\n", surfaceExtent.width, surfaceExtent.height);

  VkSurfaceFormatKHR surfaceFormat;
  VkPresentModeKHR surfacePresentMode;
  { // choose format and present mode
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());
  
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    surfaceFormat = formats[0];
    surfacePresentMode = presentModes[0];
  }

  // Queues
  VkQueue graphicsQueue = VK_NULL_HANDLE;
  uint32_t graphicsQueueIndex;
  VkQueue presentQueue = VK_NULL_HANDLE;
  uint32_t presentQueueIndex;
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
          graphicsQueueIndex = i;
        }
      }
      if(presentQueueFound == 0){
        VkBool32 present = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &present);
        if(present){
          presentQueueFound = 1;
          presentQueueIndex = i;
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
    sameQueue = graphicsQueueIndex == presentQueueIndex;
  }

  // Logical device
  VkDevice device;
  std::vector<uint32_t> uniqueQueueIndices;
  uniqueQueueIndices.push_back(graphicsQueueIndex);
  if(!sameQueue) uniqueQueueIndices.push_back(presentQueueIndex);
  // std::set<uint32_t> uniqueQueueIndices(graphicsQueueIndex, presentQueueIndex);
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  for(uint32_t q: uniqueQueueIndices){
    queueCreateInfos.push_back({
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
      .queueFamilyIndex = q,
      .queueCount = 1,
      .pQueuePriorities = Ptr(1.0f),
    });
  }
  
  if(vkCreateDevice(physicalDevice, Ptr(VkDeviceCreateInfo{
    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    // .pNext = NULL,
    // .flags = NULL,
    .queueCreateInfoCount = (uint32_t)queueCreateInfos.size(),
    .pQueueCreateInfos = queueCreateInfos.data(),
    // enabledLayerCount is deprecated and should not be used
    // .enabledLayerCount = ,
    // ppEnabledLayerNames is deprecated and should not be used
    // .ppEnabledLayerNames = ,
    .enabledExtensionCount = 1,
    .ppEnabledExtensionNames = Ptr("VK_KHR_swapchain"),
    .pEnabledFeatures = {},
  }), NULL, &device) != VK_SUCCESS) err("Failed to create device\n");
  vkGetDeviceQueue(device, graphicsQueueIndex, 0, &graphicsQueue);
  vkGetDeviceQueue(device, presentQueueIndex, 0, &presentQueue);

  // Swapchain
  VkSwapchainKHR swapchain;
  std::vector<VkImage> swapchainImages;
  std::vector<VkImageView> swapchainImageViews;
  if(vkCreateSwapchainKHR(device, Ptr((VkSwapchainCreateInfoKHR){
    .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
    // .pNext = NULL,
    // .flags = NULL,
    .surface = surface,
    .minImageCount = surfaceCapabilities.minImageCount,
    .imageFormat = surfaceFormat.format,
    .imageColorSpace = surfaceFormat.colorSpace,
    .imageExtent = surfaceExtent,
    .imageArrayLayers = 1,
    .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .imageSharingMode =      sameQueue ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
    .queueFamilyIndexCount = sameQueue ? (uint32_t)0               : (uint32_t)2,
    .pQueueFamilyIndices =   sameQueue ? NULL                      : std::vector(graphicsQueueIndex, presentQueueIndex).data(),
    .preTransform = surfaceCapabilities.currentTransform,
  #ifdef ANDROID
    .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, //android?
  #else
    .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
  #endif
    .presentMode = surfacePresentMode,
    .clipped = VK_TRUE,
    .oldSwapchain = VK_NULL_HANDLE,
  }), NULL, &swapchain) != VK_SUCCESS) err("Failed to create swapchain\n");

  {
    uint32_t imageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
    swapchainImages.resize(imageCount);
    swapchainImageViews.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

    for(uint32_t i = 0; i < imageCount; i++){
      if(vkCreateImageView(device, Ptr(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        // .pNext = NULL,
        // .flags = NULL,
        .image = swapchainImages[i],
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = surfaceFormat.format,
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
      }), NULL, &swapchainImageViews[i]) != VK_SUCCESS) err("Failed to create image view %d\n", i);
    }
  }

  // Pipeline :0
  VkPipeline pipeline;

  VkPipelineLayout pipelineLayout;
  if(vkCreatePipelineLayout(device, Ptr(VkPipelineLayoutCreateInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
    // .pNext = NULL,
    // .flags = NULL,
    .setLayoutCount = 0,
    .pSetLayouts = NULL,
    .pushConstantRangeCount = 0,
    .pPushConstantRanges = NULL,
  }), NULL, &pipelineLayout)) err("Failed to create pipeline layout\n");

  VkRenderPass pipelineRenderPass;
  if(vkCreateRenderPass(device, Ptr(VkRenderPassCreateInfo{
    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
    // .pNext = NULL,
    // .flags = NULL,
    .attachmentCount = 1,
    .pAttachments = Ptr(VkAttachmentDescription{
      // .flags = NULL,
      .format = surfaceFormat.format,
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
      // .flags = NULL,
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
      // .pNext = NULL,
      // .flags = NULL,
      .codeSize = ss.source.size() - 1, //null termination?
      .pCode = (uint32_t*)ss.source.data(),
    }), NULL, &module) != VK_SUCCESS) err("Failed to create shader module for \"%s\"\n", ss.name.data());
    shaderStages[i] = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
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
    // .pNext = NULL,
    // .flags = NULL,
    .stageCount = (uint32_t)shaderStages.size(),
    .pStages = shaderStages.data(),
    .pVertexInputState = Ptr(VkPipelineVertexInputStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
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
      // .pNext = NULL,
      // .flags = NULL,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
      .primitiveRestartEnable = VK_FALSE,
    }),
    .pTessellationState = NULL,
    .pViewportState = Ptr(VkPipelineViewportStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
      .viewportCount = 1,
      .pViewports = NULL,
      // .pViewports = Ptr(VkViewport{ // static viewport
      //    .x = 0,
      //    .y = 0,
      //    .width =  (float)surfaceExtent.width,
      //    .height = (float)surfaceExtent.height,
      //    .minDepth = 0,
      //    .maxDepth = 0,
      //  }),
      .scissorCount = 1,
      .pScissors = NULL,
      // .pScissors = Ptr(VkRect2D{ // static scissor
      //    .offset = {0, 0},
      //    .extent = surfaceExtent,
      //  }),
    }),
    .pRasterizationState = Ptr(VkPipelineRasterizationStateCreateInfo{
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
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
      // .pNext = NULL,
      // .flags = NULL,
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
      // .pNext = NULL,
      // .flags = NULL,
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
      // .pNext = NULL,
      // .flags = NULL,
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

  // Vertex buffer
  VkBuffer vertexBuffer;
  VkDeviceMemory vertexBufferMemory;
  Vertex vertices[3] = {
    {{0, 0}, {1, 0, 0}},
    {{1, 0}, {0, 1, 0}},
    {{0, 1}, {0, 0, 1}},
  };
  if(vkCreateBuffer(device, Ptr(VkBufferCreateInfo{
    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
    // .pNext = NULL,
    // .flags = NULL,
    .size = sizeof(vertices),
    .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
    .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    .queueFamilyIndexCount = 0,
    .pQueueFamilyIndices = NULL,
  }), NULL, &vertexBuffer) != VK_SUCCESS) err("Failed to create vertex buffer\n");

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);
  uint32_t memoryTypeIndex = 0;
  for(; memoryTypeIndex  < physicalDeviceMemProperties.memoryTypeCount; memoryTypeIndex ++){
    if((memRequirements.memoryTypeBits) & (1 << memoryTypeIndex ) && physicalDeviceMemProperties.memoryTypes[memoryTypeIndex].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){
      break;
    }
  }
  if(memoryTypeIndex == physicalDeviceMemProperties.memoryTypeCount) err("Failed to find memory for vertex buffer\n");
  vkAllocateMemory(device, Ptr(VkMemoryAllocateInfo{
    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
    .pNext = NULL,
    .allocationSize = memRequirements.size,
    .memoryTypeIndex = memoryTypeIndex,
  }), NULL, &vertexBufferMemory);
  vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
  void* data;
  vkMapMemory(device, vertexBufferMemory, 0, memRequirements.size, 0, &data);
    memcpy(data, vertices, sizeof(vertices));
  vkUnmapMemory(device, vertexBufferMemory);
  
  // Framebuffers
  std::vector<VkFramebuffer> swapchainFramebuffers(swapchainImages.size());
  // std::vector<VkFramebuffer> swapchainFramebuffers(FRAMES_IN_FLIGHT);
  for(uint32_t i = 0; i < (uint32_t)swapchainFramebuffers.size(); i++){
    if(vkCreateFramebuffer(device, Ptr(VkFramebufferCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,
      .renderPass = pipelineRenderPass,
      .attachmentCount = 1,
      .pAttachments = &swapchainImageViews[i],
      .width = surfaceExtent.width,
      .height = surfaceExtent.height,
      .layers = 1,
    }), NULL, &swapchainFramebuffers[i]) != VK_SUCCESS) err("Failed to create framebuffer %d\n", i);
  }

  // Command pool
  VkCommandPool cmdPool;
  if(vkCreateCommandPool(device, Ptr(VkCommandPoolCreateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    // .pNext = NULL,
    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    .queueFamilyIndex = graphicsQueueIndex,
  }), NULL, &cmdPool) != VK_SUCCESS) err("Failed to create command pool\n");

  // Command buffers
  std::vector<VkCommandBuffer> cmdBuffers(FRAMES_IN_FLIGHT);
  if(vkAllocateCommandBuffers(device, Ptr(VkCommandBufferAllocateInfo{
    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    // .pNext = NULL,
    .commandPool = cmdPool,
    .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    .commandBufferCount = FRAMES_IN_FLIGHT,
  }), cmdBuffers.data()) != VK_SUCCESS) err("Failed to allocate command buffers\n");

  // Semaphores
  std::vector<VkSemaphore> imageReadySemaphores(FRAMES_IN_FLIGHT);
  for(VkSemaphore& s: imageReadySemaphores){
    if(vkCreateSemaphore(device, Ptr(VkSemaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,      
    }), NULL, &s) != VK_SUCCESS) err("Failed to create semaphore\n");
  }
  std::vector<VkSemaphore> renderDoneSemaphores(swapchainImages.size());
  for(VkSemaphore& s: renderDoneSemaphores){
    if(vkCreateSemaphore(device, Ptr(VkSemaphoreCreateInfo{
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      // .pNext = NULL,
      // .flags = NULL,      
    }), NULL, &s) != VK_SUCCESS) err("Failed to create semaphore\n");
  }

  // Fences
  std::vector<VkFence> renderDoneFences(FRAMES_IN_FLIGHT);
  for(VkFence& f: renderDoneFences){
    if(vkCreateFence(device, Ptr(VkFenceCreateInfo{
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
      // .pNext = NULL,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    }), NULL, &f) != VK_SUCCESS) err("Failed to create fence\n");
  }

  // Main loop OvO
  uint32_t frameIdx = 0;
  while(appState.pollEvents()){
    VkSemaphore imageReadySemaphore = imageReadySemaphores[frameIdx];
    VkFence renderDoneFence = renderDoneFences[frameIdx];
    VkCommandBuffer cmdBuffer = cmdBuffers[frameIdx];

    vkWaitForFences(device, 1, &renderDoneFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &renderDoneFence);
    
    // if(swapchainImageState != VK_SUCCESS){
    //   printf("Idk %d\n", swapchainImageState);
    //   exit(0);
    // }
    // if(appState.surfaceSizeChanged){
    if(appState.surfaceSizeChanged){
      vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &surfaceCapabilities);
      surfaceExtent = surfaceCapabilities.currentExtent;
      if((surfaceExtent.width  == std::numeric_limits<uint32_t>::max()) ||
         (surfaceExtent.height == std::numeric_limits<uint32_t>::max())){
        surfaceExtent = appState.getSurfaceExtent();
      }
      // printf("Surface size: %dx%d\n", surfaceExtent.width, surfaceExtent.height);
      if((surfaceExtent.width == 0 || surfaceExtent.height == 0) && appState.pollEvents()){
        
      };
      appState.surfaceSizeChanged = false;

      vkDeviceWaitIdle(device);
      for(auto i: swapchainFramebuffers){
        vkDestroyFramebuffer(device, i, NULL);
      }
      for(auto i: swapchainImageViews) vkDestroyImageView(device, i, NULL);
      // for(auto i: swapchainImages) vkDestroyImage(device, i, NULL);
      
      vkDestroySwapchainKHR(device, swapchain, NULL);
      if(vkCreateSwapchainKHR(device, Ptr((VkSwapchainCreateInfoKHR){
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        // .pNext = NULL,
        // .flags = NULL,
        .surface = surface,
        .minImageCount = surfaceCapabilities.minImageCount,
        .imageFormat = surfaceFormat.format,
        .imageColorSpace = surfaceFormat.colorSpace,
        .imageExtent = surfaceExtent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode =      sameQueue ? VK_SHARING_MODE_EXCLUSIVE : VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = sameQueue ? (uint32_t)0               : (uint32_t)2,
        .pQueueFamilyIndices =   sameQueue ? NULL                      : std::vector(graphicsQueueIndex, presentQueueIndex).data(),
        .preTransform = surfaceCapabilities.currentTransform,
      #ifdef ANDROID
        .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, //android?
      #else
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      #endif
        .presentMode = surfacePresentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE,
      }), NULL, &swapchain) != VK_SUCCESS) err("Failed to recreate swapchain\n");

      {
        uint32_t imageCount;
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
        swapchainImages.resize(imageCount);
        swapchainImageViews.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchainImages.data());

        for(uint32_t i = 0; i < imageCount; i++){
          if(vkCreateImageView(device, Ptr(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            // .pNext = NULL,
            // .flags = NULL,
            .image = swapchainImages[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = surfaceFormat.format,
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
          }), NULL, &swapchainImageViews[i]) != VK_SUCCESS) err("Failed to create image view %d\n", i);
        }
      }

      for(uint32_t i = 0; i < swapchainFramebuffers.size(); i++){
        if(vkCreateFramebuffer(device, Ptr(VkFramebufferCreateInfo{
          .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
          // .pNext = NULL,
          // .flags = NULL,
          .renderPass = pipelineRenderPass,
          .attachmentCount = 1,
          .pAttachments = &swapchainImageViews[i],
          .width = surfaceExtent.width,
          .height = surfaceExtent.height,
          .layers = 1,
        }), NULL, &swapchainFramebuffers[i]) != VK_SUCCESS) err("Failed to create framebuffer %d\n", i);
      }
    }
    uint32_t swapchainIdx;
    VkResult swapchainImageState = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageReadySemaphore, VK_NULL_HANDLE, &swapchainIdx);
    if(swapchainImageState != VK_SUCCESS) appState.surfaceSizeChanged = true;
    VkFramebuffer framebuffer = swapchainFramebuffers[swapchainIdx];
    VkSemaphore renderDoneSemaphore = renderDoneSemaphores[swapchainIdx];

    vkResetCommandBuffer(cmdBuffer, 0);
    vkBeginCommandBuffer(cmdBuffer, Ptr(VkCommandBufferBeginInfo{
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      // .pNext = NULL,
      // .flags = NULL,
      .pInheritanceInfo = NULL,      
    }));
    vkCmdBeginRenderPass(cmdBuffer, Ptr(VkRenderPassBeginInfo{
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      // .pNext = NULL,
      
      .renderPass = pipelineRenderPass,
      .framebuffer = framebuffer,
      .renderArea = {
        .offset = {0, 0},
        .extent = surfaceExtent,
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
      .width = (float)(surfaceExtent.width),
      .height = (float)(surfaceExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);

    VkRect2D scissor{
      .offset = {0, 0},
      .extent = surfaceExtent,
    };
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    vkCmdBindVertexBuffers(cmdBuffer, 0, 1, &vertexBuffer, Ptr<VkDeviceSize>(0));

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmdBuffer);
    if(vkEndCommandBuffer(cmdBuffer) != VK_SUCCESS) err("Failed to end command buffer\n");

    VkResult queueSubmitResult = vkQueueSubmit(graphicsQueue, 1, Ptr(VkSubmitInfo{
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      // .pNext = NULL,
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
      // .pNext = NULL,
      
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &renderDoneSemaphore,
      
      .swapchainCount = 1,
      .pSwapchains = &swapchain,
      .pImageIndices = &swapchainIdx,
      .pResults = NULL,
    }));
    // })) != VK_SUCCESS) err("Failed to present queue\n");

    frameIdx = (frameIdx + 1)%FRAMES_IN_FLIGHT;
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
  vkDestroyBuffer(device, vertexBuffer, NULL);
  vkFreeMemory(device, vertexBufferMemory, NULL);
  for(VkPipelineShaderStageCreateInfo& ssc: shaderStages){
    vkDestroyShaderModule(device, ssc.module, NULL);
  }
  vkDestroyPipeline(device, pipeline, NULL);
  vkDestroyRenderPass(device, pipelineRenderPass, NULL);
  vkDestroyPipelineLayout(device, pipelineLayout, NULL);
  for(VkImageView imv: swapchainImageViews){
    vkDestroyImageView(device, imv, NULL);
  }
  vkDestroySwapchainKHR(device, swapchain, NULL);
  vkDestroyDevice(device, NULL);
  vkDestroySurfaceKHR(instance, surface, NULL);
  vkDestroyInstance(instance, NULL);
}
