# apk shit
NAME := test
APKFILE := main.apk
ANDROIDTARGET := 36

all: $(APKFILE)

run: $(APKFILE)
	adb install $(APKFILE)
	adb shell am start -n com.qwas.$(NAME)/android.app.NativeActivity
	adb shell logcat -c
	adb logcat | grep "main\|$(NAME)"

ifeq ($(TARGET),)
$(error TARGET is not set)
endif
ifeq ($(ARCH),)
$(error ARCH is not set)
endif

SDK_LOCATIONS := /opt/android-sdk/

ANDROIDSDK ?= /opt/android-sdk/
NDK ?= $(firstword $(ANDROID_NDK) $(ANDROID_NDK_HOME) $(wildcard $(ANDROIDSDK)/ndk/*) $(wildcard $(ANDROIDSDK)/ndk-bundle/*) )
BUILD_TOOLS ?= $(lastword $(wildcard $(ANDROIDSDK)/build-tools/*) )

ANDROID_JAR:=$(ANDROIDSDK)platforms/android-$(ANDROIDTARGET)/android.jar

#key
ALIASNAME?=standkey
STOREPASS?=password
DNAME:="CN=example.com, OU=ID, O=Example, L=Doe, S=John, C=GB"

key.jks:
	keytool -genkey -v -keystore key.jks -alias $(ALIASNAME) -keyalg RSA -keysize 2048 -validity 10000 -storepass $(STOREPASS) -keypass $(STOREPASS) -dname $(DNAME)

AndroidManifest.xml: AndroidManifest.xml.template
	rm -rf AndroidManifest.xml
	NAME=$(NAME) ANDROIDTARGET=$(ANDROIDTARGET) envsubst '$$ANDROIDTARGET $$NAME' < AndroidManifest.xml.template > AndroidManifest.xml

clean:
	rm -rf $(APKFILE) AndroidManifest.xml

$(APKFILE): $(TARGET) AndroidManifest.xml key.jks
	rm -rf $(APKFILE)
	aapt p -f -F tmp.apk -I $(ANDROID_JAR) -M AndroidManifest.xml -S apkTemplate/res -A apkTemplate/assets --target-sdk-version $(ANDROIDTARGET)
	rm -rf tmp
	unzip -o tmp.apk -d tmp
	rm -rf tmp.apk

	mkdir tmp/lib
ifeq ($(ARCH), aarch64)
	mkdir tmp/lib/arm64-v8a
	cp $(TARGET) tmp/lib/arm64-v8a/libmain.so
	cp /opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so tmp/lib/arm64-v8a/libc++_shared.so
else
$(error ARCH=$(ARCH) not supported)
endif

	cp $(ASSETS) tmp/ -r

	cp libs/* tmp/lib/ -r

	cd tmp && zip -D0r ../tmp.apk ./*
	rm -rf tmp
	zipalign -v 4 -f tmp.apk tmp2.apk
	rm -rf tmp.apk
	apksigner sign --key-pass pass:$(STOREPASS) --ks-pass pass:$(STOREPASS) --min-sdk-version 21 --ks key.jks tmp2.apk 
	mv tmp2.apk $(APKFILE)
