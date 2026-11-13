# Triangle-mesh RF scene backend

`simulation.mesh` implements a usable geometry-to-reflector stage for RadSimReal's Fig. 2(a). It accepts actual triangle geometry and performs line-of-sight tests, then hands weighted reflection points to the existing 3D PSF operator. No CARLA installation, optical renderer or GPU is needed for OBJ/NPZ geometry.

The authors describe CARLA scene generation, RF ray tracing and material/orientation/range-dependent reflectivity, but do not supply their scene assets, electromagnetic parameters or complete scattering implementation. This module is an independently specified model. It does not recover those missing resources or establish sim-to-real accuracy merely by executing.

## Implemented geometry

- `MeshScene` stores world vertices, triangles, per-triangle material/object IDs and translational velocities, plus sensor position, orthogonal radar-local-to-world basis and world velocity.
- `import_obj` accepts OBJ vertices and polygon faces, slashed/negative indices, `usemtl`, object/group names and planar concave polygons through ear clipping. Optical MTL colors are not interpreted as RF material constants. Supply RF materials separately. A simple polygon must be planar and non-self-intersecting.
- `save_mesh`/`load_mesh` preserve geometry, dynamic metadata and complete material definitions without Python pickle. `export_obj` exports geometry and material names, while NPZ remains the lossless interchange format.
- `mesh_from_scene` turns procedural car footprints into closed triangle extrusions. These are still box geometries. Real vehicle geometry can instead be supplied by CAD/OBJ exports.
- `ground_plane` creates an explicitly configured static surface. `merge_meshes` combines static environment CAD, ground and moving vehicle meshes; material properties and object identity mappings are recorded.
- `ray_triangle_intersections` uses a median-split bounding-volume hierarchy and double-sided Möller–Trumbore intersections. Surface shading is one-sided unless the RF material explicitly allows both sides. Endpoint exclusion avoids self-shadowing at the sampled surface.

Returned `ReflectionPoints.xyz` uses **x right, y forward, z up** relative to the radar. `velocity` is already relative to sensor ego motion. Positions and velocities must not be transformed or ego-compensated a second time before radar formation.

Exactly zero RF coefficients are removed before returning points. Metadata
separates geometrically visible surface samples from nonzero return counts;
this is an exact-zero test, not a detector threshold or CFAR visibility claim.

## Surface sampling and RCS convention

The tracer samples triangles with probability proportional to area and samples locations uniformly within each triangle. Every sample carries the quadrature area `total_surface_area / number_of_samples`, including when the configured sample cap is reached. Splitting a triangle therefore does not create extra scattering area.

In default intensity or random-field mode, each patch gets an approximate RCS contribution:

```text
sigma_patch = patch_area * backscatter_scale * |Fresnel(theta)|²
              * cos(theta) * rough_diffuse_specular_mixture(theta)
```

Fresnel TE/TM reflection is calculated for an air/nonmagnetic isotropic interface. Complex relative permittivity uses the `exp(+j omega t)` convention:

```text
epsilon_r = epsilon_real * (1 - j*loss_tangent)
            - j*conductivity/(omega*epsilon_0)
```

The mixture is a disclosed approximation using a Phong-type monostatic lobe around the specular direction and a diffuse term. Surface-height roughness attenuates coherent power by `exp[-(4*pi*sigma_height*cos(theta)/lambda)²]` and transfers weight toward the diffuse component. It is not a full microfacet solution or calibrated automotive BRDF/RCS model. Default material numbers are illustrative. The scale, dielectric/loss values, roughness and mixture require measurements or justified material data.

Without a link budget, patch powers use the project's existing relative gain and propagation law:

```text
relative_power = scattering_gain * sigma_patch * (reference_range/range)⁴
```

With an explicit `RadarLinkBudget`, the free-space monostatic equation is used:

```text
received_watts = Pt * Gt * Gr * lambda² * sigma_patch
                 / ((4*pi)³ * range⁴ * system_loss)
```

The Watt value is conditional on the supplied gains, losses and RCS model. It is not an absolute-RCS calibration guarantee. Antenna-pattern variation, path-dependent atmospheric loss and sensor output gain must be characterized separately. Do not add range attenuation again to exported coefficients or silently mix Watts and measured PSF units.

## Phase and resolution

Intensity mode sums nonnegative patch powers; it is the existing phenomenological linear-response option. `representation: complex` with default `phase_mode='random'` returns square-root-power weights with two-way propagation phase plus an independent rough-surface phase. Expected incoherent power is stable as the sample density changes, while individual speckle realizations change. Optional exponential power speckle is separately configured.

`phase_mode='path'` is a coherent scalar physical-optics area quadrature. Patch fields scale with **area**, not square-root area, and include Fresnel phase, projected area, coherent roughness attenuation and `exp(-j*4*pi*range/lambda)`. Its normal-incidence smooth-plate limit has `sigma = 4*pi*A²/lambda²`. The diffuse component is excluded in this mode. This oscillatory integral requires sufficiently fine/wavelength-resolving sampling; coarse automotive mesh sampling can alias the phase and must not be treated as validated coherent ray tracing. It is an opt-in diagnostic/extension, not the default rough-surface model.

These modes have different coherence assumptions and need independently consistent PSF/reflectivity calibration. A power-only measured PSF does not provide the phase response required for a coherent sensor model.

## Example

```python
import numpy as np
from radsimreal.config import RadarConfig
from radsimreal.simulation.mesh import import_obj, trace_reflections
from radsimreal.radar import simulate

config = RadarConfig(representation="complex")
mesh = import_obj("scene.obj")
points = trace_reflections(mesh, config, rng=np.random.default_rng(7),
                          samples_per_square_meter=8, max_samples=20000)
result = simulate(points, config, rng=np.random.default_rng(8))
```

For training labels, preserve original scene/object annotations. Imported OBJ group names do not independently establish class, footprint, orientation or benchmark ground truth.

## Moving supplied vehicle assets

`simulation.assets` supports actual user-supplied OBJ vehicle templates in moving scenes. Geometry is loaded once, then independently placed for each scene vehicle. An example manifest is:

```json
{
  "coordinate_system": "x_right_y_forward_z_up",
  "fit_dimensions": true,
  "height": 1.6,
  "assets": [
    {
      "obj": "meshes/vehicle.obj",
      "material_overrides": {
        "body": {"relative_permittivity_real": 4.0, "loss_tangent": 0.02},
        "glass": {"relative_permittivity_real": 2.25, "roughness_m": 0.0001}
      }
    }
  ]
}
```

These numerical RF values are illustrative; they are not calibrated automotive material data. Relative OBJ paths resolve against the manifest. `usemtl` names select the supplied RF overrides. Unknown materials retain an explicit uncalibrated default; optical MTL data are ignored. OBJ group/object names remain in source provenance, while whole placed vehicles receive their actual scene object IDs for native annotations.

```python
from radsimreal.simulation.assets import load_vehicle_assets, mesh_from_assets

library = load_vehicle_assets("vehicle_assets.json")
dynamic_mesh = mesh_from_assets(scene, library, rng)
points = trace_reflections(dynamic_mesh, config.radar, rng=rng)
```

The loader requires nondegenerate lateral/forward/vertical bounds. Assets must already use declared x-right/y-forward/z-up axes; bounding shape cannot prove which end is the vehicle's nose. Positive per-axis scaling fits the template bounds to each vehicle's physical width, length and the declared height, preserving winding and source topology. The footprint is centered at the vehicle position, the lowest vertex is grounded at z=0, and placement follows the existing clockwise-from-forward yaw convention. Face velocities carry the vehicle translation; ego motion is handled by the normal tracing path.

Setting `fit_dimensions:false` preserves source units, but the requested vehicle dimensions and height must already match its source bounds within `dimension_tolerance` (default 1e-5). A mismatch raises an error instead of allowing inaccurate ground-truth boxes. Source OBJ and manifest SHA256 hashes, source extents, scale, pose, velocity, material names and selection index are saved in the mesh metadata. Template selection is deterministic for a given NumPy generator seed. Templates are not modified by placement.

This supplies dynamic CAD support without claiming possession of the authors' CARLA meshes. Actual source geometry, material quality and RF scattering calibration still determine fidelity.

## Read-only CARLA snapshot export

`simulation.carla_bridge.snapshot_from_carla` lazily imports a locally installed matching CARLA client. It reads one world snapshot, actor transforms/velocities and public bounding boxes; it does not tick, spawn/delete actors, modify weather or change world settings. Optional static level boxes can be selected by `CityObjectLabel`.

```sh
python -m radsimreal.simulation.carla_bridge --host localhost --port 2000 --sensor-actor-id 123 --out runs/carla_snapshot.npz
```

The public bounding-box API does not return the authors' RF mesh/material assets. The exported geometry is therefore explicitly **triangulated bounding boxes**. Supply true CAD meshes using OBJ/NPZ when needed. CARLA's forward/right axis convention is converted to the project's right/forward basis. Snapshot actor states are preferred; any live-state fallback is recorded. The selected ego/sensor actor is excluded from the box occluders, so ego-body masking requires separate geometry.

Actual connection to a CARLA server has not been executed in the local test environment. The offline box conversion and optional-dependency behavior are tested. Neither that test nor a CARLA visual scene establishes radar realism.

## What remains outside this backend

Multiple-bounce propagation, transmission through materials, diffraction, polarization transport between bounces, full electromagnetic vehicle resonances and weather scattering are not implemented. They require additional physical models and validation; the code does not silently synthesize them. The PSF remains the calibrated receiver/image-formation interface. Material calibration and real-data comparisons on the user's cluster are still required.

## Sources and checks

- [RadSimReal Section 3 and Appendix B](https://arxiv.org/html/2404.18150v1): environment/reflection-point stage followed by PSF image formation.
- [MIT electromagnetics course notes, radar equation, Section 11.4](https://ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/d3be4ea78b036a6362230fb41780cf54_MIT6_013S09_notes.pdf), and [MathWorks radar-equation documentation](https://www.mathworks.com/help/radar/ug/radar-equation.html): link-budget convention.
- [Ellingson's TE planar-interface derivation](https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Electromagnetics_II_(Ellingson)/05%3A_Wave_Reflection_and_Transmission/5.06%3A_Plane_Waves_at_Oblique_Incidence_on_a_Planar_Boundary-_TE_Case) and [TM derivation](https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Electromagnetics_II_(Ellingson)/05%3A_Wave_Reflection_and_Transmission/5.07%3A_Plane_Waves_at_Oblique_Incidence_on_a_Planar_Boundary-_TM_Case): Fresnel boundary conditions. The chosen rough-scattering mixture is explicitly additional modeling.
- [Official CARLA bounding-box tutorial](https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/) and [API reference](https://carla.readthedocs.io/en/latest/python_api/): actor/level bounding boxes, snapshot states and transforms.

Automated tests check normal-incidence/Brewster Fresnel values, passive response, the monostatic range law, triangle area/visibility, near-surface endpoint handling, occlusion, sensor pose/relative velocity, phase, surface quadrature, planar OBJ triangulation, NPZ/OBJ interchange, degenerate input and the offline CARLA bridge. These are mathematical/software checks, not measured radar fidelity results.
