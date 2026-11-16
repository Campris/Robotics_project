"""
Geometria, respiro, eventi e modello di contatto Hunt-Crossley del torso.

La superficie e' analitica z = zs(x, y, t) (geometria + respiro + eventi).
Tissue.contact() calcola la forza del tessuto sulla sonda (Hunt-Crossley): e' QUESTA la forza
che il controllore di ammettenza applica al robot (panda_scene.apply_tip_force).

Convenzioni: unita' SI. z verso l'alto, la normale n punta fuori dal corpo.
delta > 0 = penetrazione della punta nel tessuto.
"""
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def min_jerk(s):
    """0 -> 1 con derivata prima e seconda nulle agli estremi (C2)."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def min_jerk_dot(s):
    """Derivata di min_jerk rispetto a s."""
    s = np.clip(s, 0.0, 1.0)
    return 30 * s**2 - 60 * s**3 + 30 * s**4


@dataclass
class TorsoGeometry:
    """Sezione ellittica (curvatura trasversale) con asse x cranio-caudale."""
    H: float = 0.10
    B: float = 0.20
    taper: float = 0.0
    z_base: float = 0.0

    def z0(self, x, y):
        r = np.clip(y / self.B, -0.9, 0.9)
        return self.z_base + self.H * np.sqrt(1.0 - r**2) * (1.0 + self.taper * x)


@dataclass
class StiffnessMap:
    """
    Rigidita' EQUIVALENTE k_eq(x, y) [N/m] alla profondita' di riferimento
    (pendenza dF/d(delta) a delta_ref, rigidezza tangente). Torace (x < x_tr): piu' rigido, con costole
    e sterno. Addome (x > x_tr): morbido. Tutto liscio, senza gradini.
    """
    k_abdomen: float = 500.0
    k_chest: float = 900.0
    k_rib: float = 600.0
    k_sternum: float = 700.0
    x_tr: float = 0.02
    rib_pitch: float = 0.03

    def __call__(self, x, y):
        w_th = 1.0 - sigmoid((x - self.x_tr) / 0.02)
        ribs = (0.5 * (1 + np.cos(2 * np.pi * x / self.rib_pitch))) ** 2
        ribs *= sigmoid((abs(y) - 0.03) / 0.01)         
        sternum = np.exp(-0.5 * (y / 0.015) ** 2)
        return (self.k_abdomen
                + w_th * ((self.k_chest - self.k_abdomen)
                          + self.k_rib * ribs + self.k_sternum * sternum))


@dataclass
class Breathing:
    """
    u(x, t) = gain(x) * s(t) [m]: gain = QUANTO sale quel punto, s = QUANDO (0 espirato, 1 inspirato, max 1.15).
    Con fn = la TUA funzione t -> s si sostituisce il modello di s() (period viene ignorato).
    """
    period: float = 4.0
    amp_chest: float = 0.0015
    amp_abdomen: float = 0.0055
    x_tr: float = 0.02
    fn: Optional[Callable[[float], float]] = None

    def s(self, t):
        if self.fn is not None:
            return self.fn(t)
        phi = 2 * np.pi * t / self.period
        phi_w = phi + 0.35 * np.sin(phi)
        mod = 1.0 + 0.15 * np.sin(2 * np.pi * t / 17.0)
        return mod * 0.5 * (1.0 - np.cos(phi_w))

    def gain(self, x):
        return self.amp_chest + (self.amp_abdomen - self.amp_chest) * sigmoid((x - self.x_tr) / 0.03)

    def u(self, x, t):
        return self.gain(x) * self.s(t)


@dataclass
class Event:
    """Evento liscio. 'gauss' = impulso gaussiano (t0 = picco, duration = sigma); 'bump' sale e scende (C2); 'step' resta."""
    t0: float
    duration: float
    amplitude: float
    kind: str = "bump"

    def value(self, t):
        if self.kind == "gauss":
            return self.amplitude * np.exp(-0.5 * ((t - self.t0) / self.duration) ** 2)
        s = (t - self.t0) / self.duration
        if self.kind == "step":
            return self.amplitude * min_jerk(s)   
        if s <= 0.0 or s >= 1.0:
            return 0.0
        return self.amplitude * min_jerk(1.0 - abs(2.0 * s - 1.0))


def gaussian_events(seed=0, t_end=200.0, t_first=10.0, rate=1 / 20, A=(0.005, 0.010), sigma=(0.15, 0.35)):
    """Sussulti casuali (appunti T2): impulsi gaussiani con picco t0, ampiezza A [m] e sigma [s] estratti a caso."""
    rng = np.random.default_rng(seed)
    events = []
    t0 = t_first + rng.exponential(1 / rate)
    while t0 < t_end:
        events.append(Event(t0, rng.uniform(*sigma), rng.uniform(*A), "gauss"))
        t0 += rng.exponential(1 / rate)
    return events


@dataclass
class ContactResult:
    force: np.ndarray
    fn: float
    delta: float
    ddelta: float
    k_eq: float
    normal: np.ndarray
    zs: float


class Tissue:
    """
    Contatto Hunt-Crossley: F = k * d^n * (1 + 1.5 * alpha * d_dot)
      k      [N/m^n]  derivato da k_eq(x,y): k = k_eq / (n * delta_ref^(n-1))  (dF/d(delta) = k_eq a delta_ref)
      n      1.5 = Hertz (rigidita' che cresce con la penetrazione)
      alpha  [s/m]  dissipazione
    Anti-spike:
      * penetrazione liscia (softplus, scala eps) -> nessun gradino all'inizio del contatto
      * il fattore di smorzamento e' saturato a >= 0 -> la forza non diventa mai negativa
      * il termine viscoso e' proporzionale a d^n -> va a zero con continuita' al distacco
      * respiro/eventi C2 -> accelerazione limitata della superficie
      * nel calcolo di d_dot entra anche lo scorrimento lungo la superficie curva
    """

    def __init__(self, geom=None, stiff=None, breathing=None, events=None,
                 n=1.5, alpha=1.5, delta_ref=0.008, eps=1e-4):
        self.geom = geom or TorsoGeometry()
        self.stiff = stiff or StiffnessMap()
        self.breathing = breathing or Breathing()
        self.events = list(events or [])
        self.n, self.alpha, self.delta_ref, self.eps = n, alpha, delta_ref, eps

    def surface_height(self, x, y, t):
        z = self.geom.z0(x, y) + self.breathing.u(x, t)
        for e in self.events:
            z += e.value(t)
        return z

    def _grad(self, x, y, t, h=1e-4):
        gx = (self.surface_height(x + h, y, t) - self.surface_height(x - h, y, t)) / (2 * h)
        gy = (self.surface_height(x, y + h, t) - self.surface_height(x, y - h, t)) / (2 * h)
        return gx, gy

    def normal(self, x, y, t):
        gx, gy = self._grad(x, y, t)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)

    def contact(self, p, v, t, h=5e-4) -> ContactResult:
        x, y, zp = p
        zs = self.surface_height(x, y, t)
        gx, gy = self._grad(x, y, t)
        nvec = np.array([-gx, -gy, 1.0])
        inv_norm = 1.0 / np.linalg.norm(nvec)
        n_hat = nvec * inv_norm
        nz = n_hat[2]

        delta_raw = (zs - zp) * nz
        zs_dot = (self.surface_height(x, y, t + h) - self.surface_height(x, y, t - h)) / (2 * h)
        v_pelle = np.array([0.0, 0.0, zs_dot])
        ddelta_raw = np.dot(v_pelle - v, n_hat)

        d = self.eps * np.logaddexp(0.0, delta_raw / self.eps)
        dd = sigmoid(delta_raw / self.eps) * ddelta_raw

        k_eq = float(self.stiff(x, y))
        k = k_eq / (self.n * self.delta_ref ** (self.n - 1.0))
        damping_factor = max(0.0, 1.0 + 1.5 * self.alpha * dd)
        fn = k * d**self.n * damping_factor
        return ContactResult(fn * n_hat, fn, delta_raw, ddelta_raw, k_eq, n_hat, zs)
