"""
Esegue il controllo di ammettenza (admittance_control.py) e produce grafici e CSV a 1 kHz.

  python run_admittance.py                       # discesa + scansione a SERPENTINA su tutto il torace (durata automatica)
  python run_admittance.py --scan-pattern line   # una sola linea (SCAN_START -> SCAN_END)
  python run_admittance.py --no-scan             # solo discesa e mantenimento della forza (15 s)
  python run_admittance.py --viewer              # finestra MuJoCo interattiva
  python run_admittance.py --plot-schedule       # disegna K_d(d), forza di riferimento e limite di velocita'

  # percorso e velocita' della serpentina
  python run_admittance.py --scan-x -0.16 0.16 --scan-y -0.11 0.11 --row-spacing 0.04 --scan-speed 0.02
  python run_admittance.py --duration 60         # taglia la simulazione a 60 s (utile per prove veloci)

  # sensore di forza (default: ATI Nano17 SI-25-0.25 + Net F/T senza filtro, vedi ft_sensor.py)
  python run_admittance.py --ati-cal SI-12-0.12 --ati-filter 152

  # controllore
  python run_admittance.py --M 0.05 --fdes 5 --integrator implicit --kt-mode k_eq --events-seed 3
  python run_admittance.py --kfar 1000 --knear 200 --dfar 0.08 --dnear 0.03 --doff 0.0

Il CSV ha un campione ogni --csv-stride passi (default 1 = 1 kHz; con 170 s sono circa 30 MB: usa 10 per 100 Hz).
"""
import argparse
import csv
import time
from pathlib import Path
import numpy as np

import panda_scene as ps
from admittance_control import AdmittanceParams, AdmittanceSim, StiffnessSchedule, GOAL_DEPTH
from tissue import gaussian_events
from ft_sensor import CALIBRATIONS, NETFT_FILTERS

FS = 1000


def make_params(a):
    p = AdmittanceParams()
    for arg, field in (("fdes", "F_DES"), ("M", "M_D"), ("kfar", "K_FAR"), ("knear", "K_NEAR"), ("dfar", "D_FAR"),
                       ("dnear", "D_NEAR"), ("doff", "D_OFF"), ("integrator", "INTEGRATOR"), ("kt_mode", "KT_MODE"),
                       ("scan_pattern", "SCAN_PATTERN"), ("row_spacing", "SCAN_ROW_SPACING"),
                       ("scan_speed", "SCAN_SPEED"), ("scan_delay", "SCAN_DELAY")):
        v = getattr(a, arg)
        if v is not None:
            setattr(p, field, v)
    for arg, field in (("scan_x", "SCAN_X_RANGE"), ("scan_y", "SCAN_Y_RANGE"),
                       ("scan_start", "SCAN_START"), ("scan_end", "SCAN_END")):
        v = getattr(a, arg)
        if v:
            setattr(p, field, tuple(v))
    if a.b_include_kd:
        p.B_INCLUDE_KD = True
    if a.no_scan:
        p.SCAN_ENABLE = False
    return p


def build_sim(a):
    sim = AdmittanceSim(make_params(a), descent_time=a.descent_time, goal_depth=a.goal_depth,
                        ati_calibration=a.ati_cal,
                        ati_filter=None if a.ati_filter == "none" else int(a.ati_filter))
    print(f"Sensore: ATI Nano17 {a.ati_cal}, filtro Net F/T {a.ati_filter}, "
          f"rumore (ipotesi) {sim.ft.noise_std[2]*1000:.1f} mN, portata Fz {sim.ft.range[2]:g} N")
    if sim.p.SCAN_ENABLE:
        estimated = a.descent_time + sim.p.SCAN_DELAY + sim.traj.T
        print(f"Scansione {sim.p.SCAN_PATTERN}: {len(sim.traj.waypoints)} waypoint, "
              f"{sim.traj.length:.2f} m, circa {estimated:.0f} s con contatto continuo")
    return sim


def run_viewer(sim, duration, vis_gain=1.0):
    """Finestra MuJoCo interattiva: ~16 passi fisici (16 ms) per fotogramma, in tempo reale."""
    import mujoco.viewer
    m, d, ids = sim.model, sim.data, sim.ids
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 140, -25, 1.3
        v.cam.lookat[:] = [0.0, 0.0, 0.15]
        next_print = 0.0
        while v.is_running() and d.time < duration:
            t_wall = time.time()
            for _ in range(16):
                fn = sim.step()
            comp = ps.measure_tissue_compression(m, d, ids, sim.tissue, d.time)
            centre = d.geom_xpos[ids["tip_geom"]]
            ps.update_tissue_mesh(m, sim.tissue, d.time, vis_gain, (centre[0], centre[1], comp) if comp > 0 else None)
            green, red = abs(fn - sim.p.F_DES) <= 1.0, fn > 15.0
            m.geom_rgba[ids["tip_geom"]] = (0.2, 0.8, 0.3, 1) if green else ((0.9, 0.2, 0.2, 1) if red else (0.95, 0.8, 0.2, 1))
            v.update_hfield(ids["hfield"])
            v.sync()
            if d.time >= next_print:
                L = sim.log
                print(f"t={d.time:6.2f}s  F vera {fn:5.2f} N  F sensore {L['fmeas'][-1]:5.2f} N  "
                      f"penetrazione {L['delta'][-1] * 1000:5.2f} mm  K_d {L['Kd'][-1]:6.0f} N/m  K_t {L['Kt'][-1]:5.0f} N/m")
                next_print += 0.5
            time.sleep(max(0.0, 0.016 - (time.time() - t_wall)))


def plot_schedule(p, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sch = StiffnessSchedule(p)
    d = np.linspace(-0.010, p.D_FAR + 0.03, 600)
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(d * 1000, [sch.K(x, 0.0) for x in d]); ax[0].set_ylabel("K_d [N/m]")
    ax[0].set_title("Soglie di K_d: alta lontano, scende vicino, si spegne a contatto (senza forza misurata)")
    ax[1].plot(d * 1000, [sch.rho(x, 0.0) * p.F_DES for x in d], color="tab:green")
    ax[1].set_ylabel("forza di riferimento attiva [N]")
    ax[2].plot(d * 1000, [sch.v_in_max(x, 0.0) * 1000 for x in d], color="tab:red")
    ax[2].set_ylabel("v massima verso la pelle [mm/s]"); ax[2].set_xlabel("distanza d dalla pelle [mm]  (>0 = sopra)")
    for a_ in ax:
        for x in (p.D_OFF, p.D_NEAR, p.D_FAR):
            a_.axvline(x * 1000, color="grey", ls=":", lw=1)
        a_.grid(alpha=0.3)
    for x, lab in ((p.D_OFF, "D_OFF"), (p.D_NEAR, "D_NEAR"), (p.D_FAR, "D_FAR")):
        ax[0].text(x * 1000, ax[0].get_ylim()[1] * 0.97, " " + lab, fontsize=8, va="top")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    print("salvato", out)


def analyse_and_plot(sim, a, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = sim.p
    L = {k: np.asarray(v) for k, v in sim.log.items()}
    t = L["t"]
    assert abs(np.diff(t).mean() - 1.0 / FS) < 1e-9
    fm = L["fmeas"]
    eF = p.F_DES - fm
    contact = L["ncontact"] > 0
    if not contact.any():
        print("ATTENZIONE: nessun contatto nella simulazione. Controlla raggiungibilita' e durata.")
        return None
    t_c = t[contact][0]
    win = t >= t_c + a.settle
    dfdt = np.abs(np.diff(L["fn"][win])) * FS
    stats = dict(
        durata_s=t[-1], primo_contatto_s=t_c, finestra=f"{t_c + a.settle:.1f}-{t[-1]:.1f} s",
        F_media=fm[win].mean(), F_min=fm[win].min(), F_max=fm[win].max(),
        picco_sopra_Fdes=max(0.0, fm[t >= t_c].max() - p.F_DES),
        eF_media=eF[win].mean(), eF_rms=np.sqrt(np.mean(eF[win] ** 2)), eF_max=np.abs(eF[win]).max(),
        entro_1N=100 * np.mean(np.abs(eF[win]) < 1.0), entro_2N=100 * np.mean(np.abs(eF[win]) < 2.0),
        max_dFdt=dfdt.max(), Kd_a_regime=L["Kd"][win].max(), Bd_medio=L["Bd"][win].mean(),
        Kt_min=L["Kt"][win].min(), Kt_max=L["Kt"][win].max(),
    )
    scan = np.zeros_like(t, bool)
    if p.SCAN_ENABLE:
        scan = (L["scan_s"] > 0.001) & (L["scan_s"] < 0.999) & win
        if scan.any():
            stats.update(scansione_s=scan.sum() / FS, scan_eF_rms=np.sqrt(np.mean(eF[scan] ** 2)),
                         scan_entro_1N=100 * np.mean(np.abs(eF[scan]) < 1.0), scan_eF_max=np.abs(eF[scan]).max(),
                         scan_exy_medio_mm=1000 * L["exy"][scan].mean(), scan_exy_max_mm=1000 * L["exy"][scan].max(),
                         scan_completata_pct=100 * L["scan_s"][-1],
                         perdite_contatto_s=float(np.sum(L["fmeas"][win] < p.F_CONTACT_OK) / FS))
    print(f"\n=== ammettenza | M_d {p.M_D} kg | F_des {p.F_DES} N | {p.INTEGRATOR} | K_t {p.KT_MODE} ===")
    for k, v in stats.items():
        print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")

    fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    ax[0].plot(t, L["fn"], lw=0.8, alpha=0.6, label="forza vera")
    ax[0].plot(t, fm, lw=0.5, color="tab:orange", label="F misurata (sensore, 1 kHz)")
    ax[0].axhline(p.F_DES, color="g"); ax[0].axhspan(p.F_DES - 1, p.F_DES + 1, color="g", alpha=0.12)
    ax[0].set_ylabel("F normale [N]"); ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    ax[1].plot(t, eF, lw=0.5, color="tab:red"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhspan(-1, 1, color="g", alpha=0.12); ax[1].set_ylabel("e_F = F_des - F_meas [N]")
    ax[1].text(0.995, 0.93, f"RMS {stats['eF_rms']:.2f} N | entro 1 N: {stats['entro_1N']:.0f}% | picco sopra F_des: "
               f"{stats['picco_sopra_Fdes']:.2f} N", transform=ax[1].transAxes, ha="right", va="top", fontsize=9)
    ax[2].plot(t, L["Kd"], color="tab:blue"); ax[2].set_ylabel("K_d [N/m]", color="tab:blue")
    a2 = ax[2].twinx(); a2.plot(t, L["Bd"], color="tab:purple", lw=0.8); a2.set_ylabel("B_d [Ns/m]", color="tab:purple")
    ax[3].plot(t, L["v_in"] * 1000, label="v verso la pelle", color="tab:green", lw=0.6)
    ax[3].plot(t, L["v_lim"] * 1000, ls="--", color="tab:red", label="limite")
    ax[3].set_ylabel("v [mm/s]"); ax[3].legend(loc="upper right", fontsize=8)
    a3 = ax[3].twinx(); a3.plot(t, L["dist"] * 1000, color="grey", lw=0.6)
    a3.set_ylabel("distanza dalla pelle [mm]", color="grey"); a3.set_ylim(-40, 100)
    ax[4].plot(t, L["zs"] * 1000, label="superficie sotto la sonda", lw=0.8)
    ax[4].plot(t, L["ztip"] * 1000, label="punta sonda", lw=0.8)
    ax[4].set_ylabel("z [mm]"); ax[4].legend(loc="upper right", fontsize=8)
    ax[5].plot(t, L["x"] * 1000, label="x punta", lw=0.8); ax[5].plot(t, L["y"] * 1000, label="y punta", lw=0.8)
    ax[5].plot(t, L["x_ref"] * 1000, "--", lw=0.6, color="tab:blue"); ax[5].plot(t, L["y_ref"] * 1000, "--", lw=0.6, color="tab:orange")
    ax[5].set_ylabel("posizione [mm]"); ax[5].legend(loc="upper right", fontsize=8); ax[5].set_xlabel("t [s]")
    a5 = ax[5].twinx(); a5.plot(t, L["Kt"], color="tab:brown", lw=0.5); a5.set_ylabel("K_t del tessuto [N/m]", color="tab:brown")
    for x in ax:
        x.grid(alpha=0.3)
        x.axvspan(t_c + a.settle, t[-1], color="grey", alpha=0.05)
    fig.suptitle(f"Ammettenza a integratore puro: F_des = {p.F_DES:g} N, M_d = {p.M_D:g} kg, {p.INTEGRATOR}, K_t {p.KT_MODE}")
    plt.tight_layout(); plt.savefig(outdir / "admittance_run.png", dpi=110); plt.close(fig)

    sub = slice(None, None, 20)
    x_mm, y_mm = L["x"][sub] * 1000, L["y"][sub] * 1000
    fig, bx = plt.subplots(1, 3, figsize=(17, 5.8))
    pts = bx[0].scatter(x_mm, y_mm, c=t[sub], s=3, cmap="viridis")
    bx[0].plot(L["x_ref"] * 1000, L["y_ref"] * 1000, "--", color="tab:orange", lw=0.8, label="percorso richiesto")
    bx[0].set_title("Percorso della sonda"); fig.colorbar(pts, ax=bx[0], label="tempo [s]"); bx[0].legend(fontsize=8)
    m2 = win[sub]
    pts = bx[1].scatter(x_mm[m2], y_mm[m2], c=np.abs(eF[sub][m2]), s=3, cmap="magma_r", vmin=0, vmax=max(1.5, 1.5 * stats["eF_rms"]))
    bx[1].set_title("Errore di forza |F_des - F_meas| sul torace"); fig.colorbar(pts, ax=bx[1], label="[N]")
    pts = bx[2].scatter(x_mm[m2], y_mm[m2], c=L["Kt"][sub][m2], s=3, cmap="viridis")
    bx[2].set_title("Rigidita' del tessuto K_t sotto la sonda"); fig.colorbar(pts, ax=bx[2], label="[N/m]")
    for b in bx:
        b.set(xlabel="x cranio-caudale [mm]", ylabel="y trasversale [mm]"); b.set_aspect("equal", adjustable="box"); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "admittance_scan_path.png", dpi=120); plt.close(fig)

    stride = max(1, a.csv_stride)
    with (outdir / "admittance_run.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "F_true_N", "F_meas_N", "F_des_N", "e_F_N", "K_d", "B_d", "K_t", "v_in_mm_s", "dist_mm",
                    "z_surface_mm", "z_tip_mm", "x_mm", "y_mm", "x_ref_mm", "y_ref_mm", "scan_progress"])
        for i in range(0, len(t), stride):
            w.writerow([f"{t[i]:.3f}", f"{L['fn'][i]:.4f}", f"{fm[i]:.4f}", p.F_DES, f"{eF[i]:.4f}", f"{L['Kd'][i]:.2f}",
                        f"{L['Bd'][i]:.2f}", f"{L['Kt'][i]:.1f}", f"{L['v_in'][i] * 1000:.3f}", f"{L['dist'][i] * 1000:.3f}",
                        f"{L['zs'][i] * 1000:.3f}", f"{L['ztip'][i] * 1000:.3f}", f"{L['x'][i] * 1000:.2f}", f"{L['y'][i] * 1000:.2f}",
                        f"{L['x_ref'][i] * 1000:.2f}", f"{L['y_ref'][i] * 1000:.2f}", f"{L['scan_s'][i]:.5f}"])
    print(f"salvati {outdir}/admittance_run.png, admittance_scan_path.png e admittance_run.csv (1 campione ogni {stride} ms)")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=None, help="default: automatica (discesa + attesa + percorso + 3 s)")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--vis-gain", type=float, default=1.0)
    ap.add_argument("--plot-schedule", action="store_true")
    ap.add_argument("--ati-cal", choices=sorted(CALIBRATIONS), default="SI-25-0.25", help="calibrazione Nano17")
    ap.add_argument("--ati-filter", choices=["none"] + [str(f) for f in NETFT_FILTERS if f], default="none",
                    help="filtro passa-basso del Net F/T [Hz]")
    ap.add_argument("--events-seed", type=int, default=0, help="seme dei sussulti gaussiani casuali; -1 = nessun sussulto")
    ap.add_argument("--descent-time", type=float, default=4.0)
    ap.add_argument("--goal-depth", type=float, default=GOAL_DEPTH, help="profondita' dell'obiettivo del planner [m]")
    ap.add_argument("--settle", type=float, default=3.0, help="secondi dopo il primo contatto prima delle statistiche")
    ap.add_argument("--csv-stride", type=int, default=1, help="un campione CSV ogni N passi (1 = 1 kHz)")
    for name in ("fdes", "M", "kfar", "knear", "dfar", "dnear", "doff"):
        ap.add_argument(f"--{name}", type=float, default=None)
    ap.add_argument("--integrator", choices=("explicit", "implicit"), default=None)
    ap.add_argument("--kt-mode", choices=("k_eq", "tangent"), default=None)
    ap.add_argument("--b-include-kd", action="store_true")
    ap.add_argument("--no-scan", action="store_true", help="niente traiettoria XY")
    ap.add_argument("--scan-pattern", choices=("serpentine", "line"), default=None)
    ap.add_argument("--scan-x", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="limiti cranio-caudali della serpentina [m]")
    ap.add_argument("--scan-y", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="limiti trasversali della serpentina [m]")
    ap.add_argument("--row-spacing", type=float, default=None, help="distanza massima tra le passate [m]")
    ap.add_argument("--scan-start", type=float, nargs=2, metavar=("X", "Y"), help="inizio della linea singola")
    ap.add_argument("--scan-end", type=float, nargs=2, metavar=("X", "Y"), help="fine della linea singola")
    ap.add_argument("--scan-speed", type=float, default=None, help="velocita' media lungo il percorso [m/s]")
    ap.add_argument("--scan-delay", type=float, default=None, help="attesa dopo il primo contatto [s]")
    a = ap.parse_args()

    outdir = Path("out"); outdir.mkdir(exist_ok=True)
    if a.plot_schedule:
        plot_schedule(make_params(a), outdir / "k_schedule.png")
    sim = build_sim(a)
    if a.duration is None:
        a.duration = (a.descent_time + sim.p.SCAN_DELAY + sim.traj.T + 3.0) if sim.p.SCAN_ENABLE else 15.0
    print(f"durata simulazione: {a.duration:.1f} s")
    if a.events_seed >= 0:
        sim.tissue.events = gaussian_events(seed=a.events_seed, t_end=a.duration)
        print("sussulti gaussiani: " + ", ".join(f"{e.t0:.1f} s ({e.amplitude*1000:.1f} mm, sigma {e.duration:.2f} s)"
                                                  for e in sim.tissue.events))
    if a.viewer:
        run_viewer(sim, a.duration, a.vis_gain)
    else:
        t0 = time.time()
        while sim.data.time < a.duration:
            sim.step()
        print(f"simulati {a.duration:.0f} s in {time.time() - t0:.0f} s di tempo reale")
        analyse_and_plot(sim, a, outdir)
