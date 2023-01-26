#version 450
#extension GL_ARB_separate_shader_objects : enable

const vec2[] vertices = {
    vec2(-.5, -.5),
    vec2(.5, .5),
    vec2(0, .5),
};

void main() {
    gl_Position = vec4(vertices[gl_VertexIndex], 0., 1.);
}