#pragma once

#include <GLFW/glfw3.h>

// mouse interaction
extern bool button_left;
extern bool button_middle;
extern bool button_right;
extern double lastx;
extern double lasty;

// keyboard callback
void keyboard(GLFWwindow *window, int key, int scancode, int act, int mods);

// mouse button callback
void mouse_button(GLFWwindow *window, int button, int act, int mods);

// mouse move callback
void mouse_move(GLFWwindow *window, double xpos, double ypos);

// scroll callback
void scroll(GLFWwindow *window, double xoffset, double yoffset);
