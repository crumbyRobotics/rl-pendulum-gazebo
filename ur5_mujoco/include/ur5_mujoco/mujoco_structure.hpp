#pragma once

#include <mujoco/mujoco.h>

// MuJoCo data structures
extern mjModel *m;     // MuJoCo model
extern mjData *d;      // MuJoCo data
extern mjvCamera cam;  // abstract camera
extern mjvOption opt;  // visualization options
extern mjvScene scn;   // abstract scene
extern mjrContext con; // custom GPU context