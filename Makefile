PLATFORM ?= linux
ARCH ?= x86_64

BUILDDIR := build/$(PLATFORM)_$(ARCH)
#final executable
TARGET := $(BUILDDIR)/target
# what the final executable needs to work
ASSETS := assets

SOURCES = main.cpp include/readFile/readFile.cpp
OBJECTS = $(patsubst %.cpp, %.o, $(patsubst %.c, %.o, $(SOURCES)))
OBJPATHS = $(patsubst %.o, $(BUILDDIR)/%.o, $(notdir $(OBJECTS)))

SHADERSOURCES = $(wildcard shaders/*.glsl)
SHADERSPVS = $(patsubst %.glsl, assets/%.spv, $(SHADERSOURCES))

fullBuild: $(TARGET) $(SHADERSPVS)
ifneq ($(wildcard platformFiles/$(PLATFORM)),)
	cd platformFiles/$(PLATFORM) && TARGET=$(realpath $(TARGET)) ASSETS=$(realpath $(ASSETS)) make
endif

run: fullBuild
ifneq ($(wildcard platformFiles/$(PLATFORM)),)
	cd platformFiles/$(PLATFORM) && TARGET=$(realpath $(TARGET)) ASSETS=$(realpath $(ASSETS)) make run
else
	./$(TARGET)
endif

listplatforms:
	@echo linux x86_64
	@echo android aarch64
	@echo windows x86_64

CC := idk
CXX := idk
CFLAGS := -Wall -I./include
CXXFLAGS = $(CFLAGS)
LDFLAGS := -lvulkan
ifeq ($(PLATFORM), linux)

ifeq ($(ARCH), x86_64)
CC := gcc
CXX := g++
LDFLAGS += -lglfw
endif

else
ifeq ($(PLATFORM), android)
CC := /opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/$(ARCH)-linux-android30-clang
CXX := $(CC)++
CFLAGS += -DANDROID -DAPPNAME=\"main\" -fPIC -m64
LDFLAGS += -L/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/30 -Wl -landroid -llog -shared -uANativeActivity_onCreate -lEGL -lGLESv1_CM
SOURCES += include/platformBridge/android_native_app_glue.c

else
ifeq ($(PLATFORM), windows)
CC := x86_64-w64-mingw32-gcc
CXX := x86_64-w64-mingw32-g++
#cringe af 
LDFLAGS += -static -lglfw
endif

endif
endif

ifeq ($(CC), idk)
$(error wrong platform or arch, see make listplatforms)
endif

moveObjs: $(OBJECTS)
	mv $(OBJECTS) $(BUILDDIR)/

$(TARGET): moveObjs
	$(CXX) $(LDFLAGS) $(OBJPATHS) -o $(TARGET) 

$(BUILDDIR):
	mkdir $(BUILDDIR) -p
$(BUILDDIR)/depend: $(TARGETS) $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -MM $(SOURCES) | sed 's|[a-zA-Z0-9_-]*\.o|$(BUILDDIR)/&|' > $(BUILDDIR)/depend
	
include $(BUILDDIR)/depend

# Shaders

assets/shaders:
	mkdir -p assets/shaders

assets/shaders/%.spv: shaders/%.glsl assets/shaders
	glslang -V $< -o $@
