/**
Defines a series of flags enabling/disabling OMP
Enable/Disable the corresponding functionality by setting the flag 
- 1 (enabled)
- 0 (disabled)
*/

// Use OMP multithreading in ShapeModel methods
#define USE_OMP_SHAPE_MODEL 1

// Use OMP multithreading in DynamicAnalysis methods
#define USE_OMP_DYNAMIC_ANALYSIS 1

// Use OMP multithreading in Ray methods
#define USE_OMP_RAY 1

// Use OMP in PC methods
#define USE_OMP_PC 1

// Use OMP in Lidar methods
#define USE_OMP_LIDAR 1

// Use OMP in ICP methods
#define USE_OMP_ICP 0

// Use OMP in ShapeFitter methods
#define USE_OMP_SHAPE_FITTER 1
