import random
import math


def random_translate(pose, max_step=1.0):
    pose.tx += random.uniform(-max_step, max_step)
    pose.ty += random.uniform(-max_step, max_step)
    pose.tz += random.uniform(-max_step, max_step)


def random_rotate(pose, max_angle_deg=15.0):
    a = math.radians(max_angle_deg)
    pose.rx += random.uniform(-a, a)
    pose.ry += random.uniform(-a, a)
    pose.rz += random.uniform(-a, a)


def random_move(pose, t_step=1.0, r_step=15.0):
    if random.random() < 0.5:
        random_translate(pose, t_step)
    else:
        random_rotate(pose, r_step)


def _unit_vector(dx, dy, dz):
    d = math.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 1e-12:
        return (0.0, 0.0, 0.0, 0.0)
    return (dx/d, dy/d, dz/d, d)


def biased_move(
    pose,
    pocket_center,
    t_step=1.0,
    r_step=15.0,
    bias_strength=0.35,
    rotate_prob=0.35,
    noise_frac=0.5,
):
    """
    A biased move that tends to translate ligand centroid towards pocket_center.

    - bias_strength: 0..1, how much of the translation step is along direction to pocket center
    - rotate_prob: probability to do rotation instead of translation
    - noise_frac: fraction of step reserved for random noise even in biased translation

    Translation:
      delta = (bias component toward center) + (random noise component)
      where |delta| ~ t_step
    """
    if pocket_center is None:
        # fallback
        return random_move(pose, t_step, r_step)

    if random.random() < rotate_prob:
        return random_rotate(pose, r_step)

    cx, cy, cz = pocket_center
    lx, ly, lz = pose.transformed_centroid()
    dx = cx - lx
    dy = cy - ly
    dz = cz - lz
    ux, uy, uz, dist = _unit_vector(dx, dy, dz)

    # If already extremely close, revert to random translation
    if dist < 1e-6 or (ux == uy == uz == 0.0):
        return random_translate(pose, t_step)

    # Split the translation budget into biased + noise parts
    bias_strength = max(0.0, min(1.0, float(bias_strength)))
    noise_frac = max(0.0, min(1.0, float(noise_frac)))

    # base magnitudes
    bias_mag = t_step * bias_strength
    noise_mag = t_step * (1.0 - bias_strength)

    # keep some guaranteed noise if requested
    noise_mag = max(noise_mag, t_step * noise_frac * 0.1)

    # biased component points toward pocket center
    bx = ux * bias_mag
    by = uy * bias_mag
    bz = uz * bias_mag

    # random noise component (uniform cube)
    nx = random.uniform(-noise_mag, noise_mag)
    ny = random.uniform(-noise_mag, noise_mag)
    nz = random.uniform(-noise_mag, noise_mag)

    pose.tx += (bx + nx)
    pose.ty += (by + ny)
    pose.tz += (bz + nz)
