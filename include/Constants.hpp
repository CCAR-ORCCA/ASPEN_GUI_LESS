// Various constants that set up the visibility emulator scenario

// HO3 Orbital elements
#define MU 1.32712440018e+20
#define ECCENTRICITY 0.10414291
#define SMA 149781865609.11
#define INCLINATION 0.135636441044066
#define HO3_RIGHT_ASCENSION 1.16087534800775
#define HO3_LONGITUDE_PERIGEE 5.36213407427001

// Central body density/mass
#define DENSITY 1900

// Lidar settings
#define ROW_RESOLUTION 256
#define COL_RESOLUTION 256
#define ROW_FOV 20
#define COL_FOV 20

// Instrument operating frequency
#define INSTRUMENT_FREQUENCY 0.0011111111111111111 // one flash every 15 minutes

// Noise
#define FOCAL_LENGTH 1e1
#define LOS_NOISE_3SD_BASELINE 0e-2

#define LOS_NOISE_FRACTION_MES_TRUTH 0.

// Times (s)
#define T0 0
// #define TF 604800// 7 days
#define TF 20000// 7 days












