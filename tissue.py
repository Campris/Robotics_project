"""
Geometria, respiro, eventi e modello di contatto Hunt-Crossley del torso.

La superficie e' analitica z = zs(x, y, t) (geometria + respiro + eventi).
Tissue.contact() conserva un modello Hunt-Crossley personalizzato; la demo attuale
usa invece la superficie come heightfield collidente e legge la forza dal solver MuJoCo.

Convenzioni: unita' SI. z verso l'alto, la normale n punta fuori dal corpo.
delta > 0 = penetrazione della punta nel tessuto.
"""
from dataclasses import dataclass, field
from collections import deque
from typing import Callable, List, Optional
import numpy as np


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def min_jerk(s):
    """0 -> 1 con derivata prima e seconda nulle agli estremi (C2)."""
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
@dataclass
class TorsoGeometry:
    """Sezione ellittica (curvatura trasversale) con asse x cranio-caudale."""
    H: float = 0.10        # altezza dell'apice [m]
    B: float = 0.20        # semiasse trasversale [m]
    taper: float = 0.0     # variazione di altezza lungo x
    z_base: float = 0.0

    def z0(self, x, y):
        r = np.clip(y / self.B, -0.9, 0.9)
        return self.z_base + self.H * np.sqrt(1.0 - r**2) * (1.0 + self.taper * x)


# --------------------------------------------------------------------------
# Rigidita' variabile con la posizione
# --------------------------------------------------------------------------
@dataclass
class StiffnessMap:
    """
    Rigidita' EQUIVALENTE lineare k_eq(x, y) [N/m] alla profondita' di riferimento
    (forza / penetrazione a delta_ref). Torace (x < x_tr): piu' rigido, con costole
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


# --------------------------------------------------------------------------
# Respiro ed eventi
# --------------------------------------------------------------------------
@dataclass
class Breathing:
    """
    Spostamento lungo z: u(x, t) = gain(x) * s(t).
    s(t) in [0, 1]. Sostituisci `fn` con la TUA funzione respiratoria t -> [0, 1].
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
        phi_w = phi + 0.35 * np.sin(phi)               # asimmetria inspiro/espiro
        mod = 1.0 + 0.15 * np.sin(2 * np.pi * t / 17.0)  # variabilita' lenta
        return mod * 0.5 * (1.0 - np.cos(phi_w))

    def gain(self, x):
        return self.amp_chest + (self.amp_abdomen - self.amp_chest) * sigmoid((x - self.x_tr) / 0.03)

    def u(self, x, t):
        return self.gain(x) * self.s(t)


@dataclass
class Event:
    """Evento liscio (C2). 'bump' sale e scende (tosse, singhiozzo); 'step' resta (spostamento)."""
    t0: float
    duration: float
    amplitude: float       # [m], positivo = la pelle sale verso la sonda
    kind: str = "bump"

    def value(self, t):
        s = (t - self.t0) / self.duration
        if self.kind == "step":
            return self.amplitude * min_jerk(s)
        if s <= 0.0 or s >= 1.0:
            return 0.0
        return self.amplitude * min_jerk(1.0 - abs(2.0 * s - 1.0))


def random_events(rng, t_end, cough_rate=1 / 20, shift_rate=1 / 40, hiccup_rate=1 / 30) -> List[Event]:
    """Eventi casuali (processi di Poisson) con ampiezze e durate realistiche."""
    events = []
    specs = [(cough_rate, "bump", (0.003, 0.006), (0.3, 0.6)),
             (hiccup_rate, "bump", (0.001, 0.003), (0.1, 0.25)),
             (shift_rate, "step", (-0.005, 0.005), (0.5, 1.5))]
    for rate, kind, amp, dur in specs:
        t = rng.exponential(1.0 / rate)
        while t < t_end:
            events.append(Event(t, rng.uniform(*dur), rng.uniform(*amp), kind))
            t += rng.exponential(1.0 / rate)
    return sorted(events, key=lambda e: e.t0)


# --------------------------------------------------------------------------
# Tessuto + contatto Hunt-Crossley
# --------------------------------------------------------------------------
@dataclass
class ContactResult:
    force: np.ndarray      # forza del tessuto sulla sonda [N], mondo
    fn: float              # modulo della forza normale [N]
    delta: float           # penetrazione [m]
    ddelta: float          # velocita' di penetrazione [m/s]
    k_eq: float            # rigidita' equivalente locale [N/m]
    normal: np.ndarray


class Tissue:
    """
    Modello opzionale Hunt-Crossley: F = k * d^n * (1 + 1.5 * alpha * d_dot)
    La demo a contatto nativo usa geometria, respiro ed eventi, ma non questa forza.
      k      [N/m^n]  derivato da k_eq(x,y): k = k_eq * delta_ref^(1-n)
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

    # -- superficie --------------------------------------------------------
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

    # -- contatto ----------------------------------------------------------
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
        ddelta_raw = (zs_dot + gx * v[0] + gy * v[1] - v[2]) * nz

        # penetrazione liscia (softplus) e sua derivata
        d = self.eps * np.logaddexp(0.0, delta_raw / self.eps)
        dd = sigmoid(delta_raw / self.eps) * ddelta_raw

        k_eq = float(self.stiff(x, y))
        k = k_eq * self.delta_ref ** (1.0 - self.n)
        damping_factor = max(0.0, 1.0 + 1.5 * self.alpha * dd)
        fn = k * d**self.n * damping_factor
        return ContactResult(fn * n_hat, fn, delta_raw, ddelta_raw, k_eq, n_hat)


# --------------------------------------------------------------------------
# Sensore di forza (modello del F_ext del Panda)
# --------------------------------------------------------------------------
class ForceSensor:
    """LPF del primo ordine + ritardo + bias + rumore. Parametri da calibrare sul robot reale."""

    def __init__(self, dt, fc=80.0, noise_std=0.15, bias=0.0, delay_s=0.003, seed=0):
        self.a = 1.0 - np.exp(-2 * np.pi * fc * dt)
        self.y = 0.0
        self.noise_std, self.bias = noise_std, bias
        self.buf = deque([0.0] * (int(round(delay_s / dt)) + 1), maxlen=int(round(delay_s / dt)) + 1)
        self.rng = np.random.default_rng(seed)

    def read(self, f_true):
        self.y += self.a * (f_true - self.y)
        self.buf.append(self.y)
        return self.buf[0] + self.bias + self.rng.normal(0.0, self.noise_std)
