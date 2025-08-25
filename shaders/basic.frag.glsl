#version 460

layout(location = 0) in vec3 color;

layout(location = 0) out vec4 fragColor;

layout(binding = 1) uniform UniformBufferObject{
    vec3 color;
} ubo;

void main() {
    fragColor = vec4(mix(color, ubo.color, 0.5), 1.0);
}
