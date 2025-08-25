#pragma once

#ifdef ANDROID
#include "android_native_app_glue.h"
#else
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#endif

#include "baseState.hpp"

void pbMain(BaseState& state);

#ifdef ANDROID
void android_main(struct android_app *app){
  BaseState state(app);
  pbMain(state);
}
#else
int main(){
  glfwInit();
  BaseState state;
  pbMain(state);
}
#endif
