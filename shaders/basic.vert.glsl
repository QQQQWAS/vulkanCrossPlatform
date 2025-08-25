#version 460

// vec2 positions[] = vec2[](
    // vec2(-0.9, -0.9),
    // vec2(0.9, -0.9),
    // vec2(-0.9, 0.9)
    // vec2(-0.9, 0.9),
    // vec2(0.9, -0.9),
    // vec2(0.9, 0.9)
// );

// vec3 colors[] = vec3[](
    // vec3(1, 0, 0),
    // vec3(0, 1, 0),
    // vec3(0, 0, 1)
// );

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

layout(location = 0) out vec3 color;

layout(binding = 0) uniform UniformBufferObject{
    mat4 transform;
} ubo;

void main() {
    color = aColor;
    gl_Position = ubo.transform * vec4(aPos, 0.0, 1.0);
}
