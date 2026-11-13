"""Material response and monostatic link budget, with explicit model boundaries.

Fresnel coefficients describe a planar air/nonmagnetic-medium interface. The
rough-surface backscatter mixture is a disclosed approximation, not a recovered
RadSimReal material model. A material's scattering scale must be calibrated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np

LIGHT_SPEED_M_S = 299792458.0
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12


@dataclass
class SurfaceMaterial:
    name: str = "uncalibrated_dielectric"
    relative_permittivity_real: float = 4.0
    loss_tangent: float = 0.02
    conductivity_s_per_m: float = 0.0
    roughness_m: float = 0.001
    diffuse_fraction: float = 0.5
    specular_exponent: float = 24.0
    backscatter_scale: float = 1.0
    two_sided: bool = False

    def __post_init__(self):
        scalars = [self.relative_permittivity_real, self.loss_tangent, self.conductivity_s_per_m,
                   self.roughness_m, self.diffuse_fraction, self.specular_exponent, self.backscatter_scale]
        if not np.isfinite(scalars).all() or self.relative_permittivity_real <= 0:
            raise ValueError("Material parameters must be finite, with positive real permittivity")
        if min(self.loss_tangent, self.conductivity_s_per_m, self.roughness_m,
               self.specular_exponent, self.backscatter_scale) < 0:
            raise ValueError("Loss, conductivity, roughness, exponent and scattering scale must be nonnegative")
        if not 0 <= self.diffuse_fraction <= 1:
            raise ValueError("diffuse_fraction must lie in [0,1]")

    def to_dict(self):
        return asdict(self)

    def complex_relative_permittivity(self, frequency_hz=77e9):
        """Return epsilon_r for exp(+j omega t), hence negative loss imaginary part."""
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        dielectric_loss = self.relative_permittivity_real * self.loss_tangent
        conduction_loss = self.conductivity_s_per_m / (2 * np.pi * frequency_hz * VACUUM_PERMITTIVITY_F_M)
        return complex(self.relative_permittivity_real, -(dielectric_loss + conduction_loss))

    def fresnel_reflection(self, cos_incidence, frequency_hz=77e9, polarization="TE"):
        """Complex field reflection for air -> isotropic nonmagnetic material.

        TE/TM bases have different reflection signs at normal incidence but
        identical reflected power. A scalar tracer keeps one chosen basis; it
        does not transport vector polarization between multiple reflections.
        """
        cosine = np.asarray(cos_incidence, dtype=float)
        if not np.isfinite(cosine).all() or np.any(cosine < 0) or np.any(cosine > 1):
            raise ValueError("cos_incidence must be finite in [0,1]")
        epsilon = self.complex_relative_permittivity(frequency_hz)
        root = np.sqrt(epsilon - (1 - cosine ** 2) + 0j)
        if polarization.upper() == "TE":
            numerator, denominator = cosine - root, cosine + root
        elif polarization.upper() == "TM":
            numerator, denominator = epsilon * cosine - root, epsilon * cosine + root
        else:
            raise ValueError("polarization must be TE or TM")
        # The matched air/air grazing limit is 0/0; reflection is zero there.
        return np.divide(numerator, denominator, out=np.zeros_like(root), where=np.abs(denominator) > 1e-30)

    def coherent_roughness_power(self, cos_incidence, frequency_hz=77e9):
        """Rayleigh roughness attenuation exp[-(4*pi*sigma*cos(theta)/lambda)^2]."""
        cosine = np.asarray(cos_incidence, dtype=float)
        if not np.isfinite(cosine).all() or np.any(cosine < 0) or np.any(cosine > 1):
            raise ValueError("cos_incidence must be finite in [0,1]")
        if not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("frequency_hz must be finite and positive")
        wavelength = LIGHT_SPEED_M_S / frequency_hz
        return np.exp(-((4 * np.pi * self.roughness_m / wavelength) * cosine) ** 2)

    def backscatter_coefficient(self, cos_incidence, frequency_hz=77e9, polarization="TE"):
        """Illustrative local differential-RCS density for incoherent patches.

        This uses Fresnel power, projected area, and diffuse/Phong-type monostatic
        lobes. The specular departure cosine is cos(2*theta). Roughness moves
        coherent weight to the diffuse component. It is nonnegative but is not
        an electromagnetic solution for a car/material microstructure.
        """
        cosine = np.asarray(cos_incidence, dtype=float)
        reflection = self.fresnel_reflection(cosine, frequency_hz, polarization)
        coherent = self.coherent_roughness_power(cosine, frequency_hz)
        specular_fraction = (1 - self.diffuse_fraction) * coherent
        lobe = np.maximum(2 * cosine ** 2 - 1, 0) ** self.specular_exponent
        directional = specular_fraction * lobe + (1 - specular_fraction)
        return self.backscatter_scale * np.abs(reflection) ** 2 * cosine * directional


@dataclass
class RadarLinkBudget:
    """Optional conditional absolute-Watt link budget; material RCS is still a model."""
    transmit_power_w: float = 1.0
    transmit_gain_linear: float = 1.0
    receive_gain_linear: float = 1.0
    system_loss_linear: float = 1.0

    def __post_init__(self):
        values = [self.transmit_power_w, self.transmit_gain_linear,
                  self.receive_gain_linear, self.system_loss_linear]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Link budget values must be finite and positive")

    def to_dict(self):
        return asdict(self)

    def power(self, rcs_m2, distance_m, frequency_hz=77e9):
        return monostatic_received_power(rcs_m2, distance_m, frequency_hz,
                                        **self.to_dict())


def monostatic_received_power(rcs_m2, distance_m, frequency_hz=77e9,
                              transmit_power_w=1.0, transmit_gain_linear=1.0,
                              receive_gain_linear=1.0, system_loss_linear=1.0):
    """Pt*Gt*Gr*lambda²*sigma / ((4*pi)³*R⁴*L), under far-field free-space assumptions."""
    sigma, distance = np.broadcast_arrays(np.asarray(rcs_m2, dtype=float), np.asarray(distance_m, dtype=float))
    values = [frequency_hz, transmit_power_w, transmit_gain_linear, receive_gain_linear, system_loss_linear]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Frequency and link budget values must be finite and positive")
    if not np.isfinite(sigma).all() or not np.isfinite(distance).all() or np.any(sigma < 0) or np.any(distance <= 0):
        raise ValueError("RCS must be finite nonnegative; distance finite positive")
    wavelength = LIGHT_SPEED_M_S / frequency_hz
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        return (transmit_power_w * transmit_gain_linear * receive_gain_linear * wavelength ** 2
                / ((4 * np.pi) ** 3 * system_loss_linear)) * sigma / distance ** 4
