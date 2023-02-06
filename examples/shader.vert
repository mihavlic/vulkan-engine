#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable

layout(scalar, binding = 0) uniform ColorOffset {
    vec3 color; float angle;
} ubo;

layout (location = 0) out vec3 outColor;

const vec2[] vertices = {
    {  1.0,  1.0 },
    { -1.0,  1.0 },
    {  0.0, -1.0 }
};

const vec3[] colors = {
    { 1.0, 0.0, 0.0 },
    { 0.0, 1.0, 0.0 },
    { 0.0, 0.0, 1.0 }
};

void main() {
    float s = sin(ubo.angle);
    float c = cos(ubo.angle);
    mat2 rot = mat2(
        vec2(c, -s),
        vec2(s, c)
    );

    gl_Position = vec4(vertices[gl_VertexIndex] * 0.75 * rot, 0., 1.);
    outColor = colors[gl_VertexIndex] + ubo.color;
}