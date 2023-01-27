#version 450
#extension GL_ARB_separate_shader_objects : enable

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
    gl_Position = vec4(vertices[gl_VertexIndex], 0., 1.);
    outColor = colors[gl_VertexIndex];
}