"""
Profili respiratori pronti. Ogni funzione restituisce un oggetto `Breathing` da passare a `Tissue`.

Regole per scrivere una tua funzione s(t):
  * t -> numero in [0, 1]  (0 = fine espirazione = superficie a riposo, 1 = fine inspirazione)
  * deve essere LISCIA (continua e con derivata continua): la velocita' della pelle entra nella
    forza (smorzamento Hunt-Crossley), quindi uno scalino in s(t) diventa uno spike di forza
  * deve essere definita per t >= 0 e coprire tutta la durata della simulazione
Il movimento reale in metri e' gain(x) * s(t), con gain = amp_chest ... amp_abdomen (vedi Breathing).

Anteprima:  python breathing_profiles.py   ->  out/breathing_profiles.png
"""
import numpy as np
from tissue import Breathing


def _cos_cycle(ph, f_in=0.4):
    """Un ciclo (ph in 0..1): inspiro per la frazione f_in, poi espiro. Parte e finisce a 0, C1."""
    if ph < f_in:
        return 0.5 * (1 - np.cos(np.pi * ph / f_in))
    return 0.5 * (1 + np.cos(np.pi * (ph - f_in) / (1 - f_in)))


def cycles(periods, amps, f_in=0.4, **kw):
    """Cicli consecutivi, ciascuno con durata e ampiezza proprie. Le liste devono coprire la simulazione."""
    starts = np.concatenate([[0.0], np.cumsum(periods)])

    def fn(t):
        t = max(t, 0.0)
        i = min(int(np.searchsorted(starts, t, side="right")) - 1, len(periods) - 1)
        return amps[i] * _cos_cycle(min((t - starts[i]) / periods[i], 1.0), f_in)
    return Breathing(fn=fn, **kw)


def regular(period=4.0, **kw):
    """Modello di default: asimmetria inspiro/espiro piu' modulazione lenta (+-15%, periodo 17 s)."""
    return Breathing(period=period, **kw)


def sinusoidal(period=4.0, **kw):
    """Coseno puro, il caso piu' semplice (utile per analizzare la risposta in frequenza)."""
    return Breathing(fn=lambda t: 0.5 * (1 - np.cos(2 * np.pi * max(t, 0.0) / period)), **kw)


def irregular(seed=0, n=80, T=(3.2, 5.0), A=(0.7, 1.15), f_in=0.4, **kw):
    """Durata e ampiezza casuali ad ogni ciclo: il caso difficile per un controllore che si aspetta periodicita'."""
    rng = np.random.default_rng(seed)
    return cycles(rng.uniform(*T, n), rng.uniform(*A, n), f_in, **kw)


def deep_slow(**kw):
    """Respiro profondo e lento (6 s, ampiezze maggiori)."""
    return cycles([6.0] * 40, [1.0] * 40, 0.4, **({"amp_chest": 0.008, "amp_abdomen": 0.020} | kw))


def shallow_fast(**kw):
    """Respiro superficiale e rapido (2 s, ampiezze ridotte)."""
    return cycles([2.0] * 120, [1.0] * 120, 0.45, **({"amp_chest": 0.003, "amp_abdomen": 0.006} | kw))


def apnea(period=4.0, hold_start=12.0, hold_dur=6.0, **kw):
    """Respiro regolare con una pausa a fine espirazione (la pausa parte al primo confine di ciclo dopo hold_start)."""
    t_h = np.ceil(hold_start / period) * period

    def fn(t):
        t = max(t, 0.0)
        tw = t if t < t_h else (t_h if t < t_h + hold_dur else t - hold_dur)
        return 0.5 * (1 - np.cos(2 * np.pi * tw / period))
    return Breathing(fn=fn, **kw)


PROFILES = {"regular": regular, "sinusoidal": sinusoidal, "irregular": irregular,
            "deep_slow": deep_slow, "shallow_fast": shallow_fast, "apnea": apnea}


if __name__ == "__main__":
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path("out").mkdir(exist_ok=True)
    t = np.linspace(0, 40, 4000)
    fig, ax = plt.subplots(len(PROFILES), 1, figsize=(10, 9), sharex=True)
    for a, (name, make) in zip(ax, PROFILES.items()):
        b = make()
        a.plot(t, [b.u(0.1, ti) * 1000 for ti in t], label="addome (x=+0.10)")
        a.plot(t, [b.u(-0.1, ti) * 1000 for ti in t], label="torace (x=-0.10)")
        a.set_ylabel("mm"); a.set_title(name, loc="left", fontsize=9); a.grid(alpha=0.3)
    ax[0].legend(fontsize=8); ax[-1].set_xlabel("t [s]")
    plt.tight_layout(); plt.savefig("out/breathing_profiles.png", dpi=100)
    print("salvato out/breathing_profiles.png")
