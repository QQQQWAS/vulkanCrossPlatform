#pragma once

#include <istream>
#include <vulkan/vulkan_core.h>
#ifdef ANDROID
  #include <vulkan/vulkan_android.h>
  #include "android_native_app_glue.h"
#else
  #define GLFW_INCLUDE_VULKAN
  #include <GLFW/glfw3.h>
#endif

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <glm/glm.hpp>
#include "../readFile/readFile.hpp"
#include "../helpers/ptr.hpp"

struct BaseState{
// private:
  glm::ivec2 surfaceSize;
  bool surfaceSizeChanged = false;
  #ifdef ANDROID
  android_app* app;

  static void onAppCmd(android_app* app, int32_t cmd){
    BaseState* state = (BaseState*)app->userData;
    switch(cmd){
      case APP_CMD_SAVE_STATE:{
        printf("App cmd APP_CMD_SAVE_STATE\n");
      }; break;
      case APP_CMD_INIT_WINDOW:{
        printf("App cmd APP_CMD_INIT_WINDOW\n");
        state->windowOpen = true;
        state->shouldExit = false;
      }; break;
      case APP_CMD_TERM_WINDOW:{
        printf("App cmd APP_CMD_TERM_WINDOW\n");
        state->windowOpen = false;
        state->shouldExit = true;
      }; break;
      case APP_CMD_GAINED_FOCUS:{
        printf("App cmd APP_CMD_GAINED_FOCUS\n");
        state->windowOpen = true;
      }; break;
      case APP_CMD_LOST_FOCUS:{
        printf("App cmd APP_CMD_LOST_FOCUS\n");
        state->windowOpen = false;
      }; break;
      case APP_CMD_CONTENT_RECT_CHANGED:
      case APP_CMD_WINDOW_RESIZED:{
        printf("App cmd APP_CMD_WINDOW_RESIZED | APP_CMD_CONTENT_RECT_CHANGED\n");
        state->surfaceSizeChanged = true;
        state->surfaceSize.x = ANativeWindow_getWidth(app->window);
        state->surfaceSize.y = ANativeWindow_getHeight(app->window);
      }; break;
      default:{
        printf("Unknown app cmd: %d\n", cmd);
      }; break;
    }
  }

  static int32_t onInputEvent(android_app* app, AInputEvent* event){
    int32_t type = AInputEvent_getType(event);
    switch(type){
      default:{
        printf("Unknown event %d\n", type);
      }; return 0;
    }
  }
  #else
  GLFWwindow* window = 0;

  static void onSizeChange(GLFWwindow* win, int x, int y){
    BaseState* bs = (BaseState*)glfwGetWindowUserPointer(win);
    bs->surfaceSize.x = x;
    bs->surfaceSize.x = y;
    if(bs->surfaceSizeChanged) printf("App does not respond to surface size changes qquickly enough\n");
    bs->surfaceSizeChanged = true;
  }
  #endif

// public:
  void (*onWindowOpened)(BaseState& state);
  void (*onWindowClosed)(BaseState& state);
  
  bool windowOpen = false;
  bool shouldExit = false;

  std::vector<const char*> vkInstanceExtensions;
  std::string getMissingInstanceExtension(){
    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

    for(auto a: vkInstanceExtensions){
      bool found = 0;
      for(auto b: availableExtensions){
        if(strcmp(a, b.extensionName) == 0){
          found = 1;
          break;
        }
      }
      if(!found){
        return a;
      }
    }
    return "";
  }
  #ifdef ANDROID
  BaseState(android_app* in_app){
    app = in_app;
    app->onAppCmd = onAppCmd;
    app->onInputEvent = onInputEvent;
    app->userData = this;
    
    vkInstanceExtensions.push_back("VK_KHR_android_surface");
    vkInstanceExtensions.push_back("VK_KHR_surface");

    std::string missing = getMissingInstanceExtension();
    if(missing != ""){
      fprintf(stderr, "Missing extension: %s\n", missing.data());
      exit(0);
    }
  }
  #else
  BaseState(){
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(640, 480, "Window", NULL, NULL);
    windowOpen = 1;
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, onSizeChange);
    
    glfwMakeContextCurrent(window);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    onSizeChange(window, w, h);
    surfaceSizeChanged = false;

    uint32_t glfwExtensionCount;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    vkInstanceExtensions.resize(glfwExtensionCount);
    for(uint32_t i = 0; i < glfwExtensionCount; i++){
      vkInstanceExtensions[i] = glfwExtensions[i];
    }

    std::string missing = getMissingInstanceExtension();
    if(missing != ""){
      fprintf(stderr, "Missing extension: %s\n", missing.data());
      exit(0);
    }
  }
  #endif
  bool pollEvents(){
    #ifdef ANDROID
    while(true){
      int events;
      android_poll_source* source;
      int pollResult = ALooper_pollOnce(0, NULL, &events, (void**)&source);
      if(pollResult < 0) break;

      if(source != NULL){
        source->process(app, source);
      }
    }
    return 1;
    #else
    glfwPollEvents();
    return !(glfwWindowShouldClose(window));
    #endif
  }

  //treat android assets as files in the "assets" dir
  #ifdef ANDROID
  std::string readFile(const char* path){
  	std::string ret = "";
    AAsset* file = AAssetManager_open(app->activity->assetManager, path, AASSET_MODE_BUFFER);
    if(file){
    	size_t fileLength = AAsset_getLength(file);
    	ret.resize(fileLength+1);
    	memcpy(ret.data(), AAsset_getBuffer(file), fileLength);
    	ret[fileLength] = 0;
    }
    else{
      printf("%s is not a valid file path\n", path);      
    }
    return ret;
  }
  #else
  std::string readFile(const char* path){
    std::string p = "assets/";
    p += path;
    return rf::readFile(p.data(), std::iostream::binary);
  }
  #endif

  VkSurfaceKHR getSurface(VkInstance instance){
    VkSurfaceKHR ret;
  #ifdef ANDROID
    while(app->window == NULL){
      pollEvents();
    }
    vkCreateAndroidSurfaceKHR(instance, Ptr((VkAndroidSurfaceCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR,
      // .pNext = NULL,
      // .flags = NULL,
      .window = app->window,
     }), NULL, &ret);
  #else
    
    glfwCreateWindowSurface(instance, window, NULL, &ret);
  #endif
    return ret;
  }
  VkExtent2D getSurfaceExtent(){
    VkExtent2D ret;
  #ifdef ANDROID
    ret.width = ANativeWindow_getWidth(app->window);
    ret.height = ANativeWindow_getHeight(app->window);
  #else
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    ret.width = w;
    ret.height = h;
  #endif
    return ret;
  }
};


