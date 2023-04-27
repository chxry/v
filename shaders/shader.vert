#version 450
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 uv;

layout(binding = 0) uniform ubo {
    mat4 model;
    mat4 view;
    mat4 proj;
} u;

void main() {
    gl_Position = u.proj * u.view * u.model * vec4(in_pos, 1.0);
    uv = in_uv;
}
