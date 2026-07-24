#pragma once

// Coarse face index currently selected for sampling (-1 = none).
extern int  gSelectedFace;

// Number of barycentric samples to generate inside the selected face.
extern int  gFaceSampleN;

// Visibility toggles for the two sample point clouds.
extern bool gShowFaceSampleCoarse;
extern bool gShowFaceSampleFine;

// Regenerate all face-sample structures (samples, hull, highlighted face).
void rebuild_face_sample_viz();

// Toggle visibility of already-registered sample point clouds without rerunning the query.
void update_face_sample_visibility();
