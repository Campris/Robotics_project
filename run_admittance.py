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
# =====================================================================================================
#  COME SONO SCRITTI I COMMENTI DI STUDIO
#    COSA   = cosa fa la riga, pezzo per pezzo
#    PERCHÉ = a cosa serve qui e cosa succederebbe senza
#    DOVE   = chi usa il risultato
#  I numeri tra parentesi sono MISURATI con "python run_admittance.py" senza opzioni (13-11-2026).
#
#  QUESTO FILE NON CONTIENE FISICA NE' CONTROLLO: legge le opzioni, costruisce AdmittanceSim, chiama step()
#  ogni ms e poi disegna/salva i segnali di sim.log. Ordine di esecuzione (in fondo, dopo "if __name__"):
#    1) argparse legge le opzioni      2) make_params -> AdmittanceParams      3) build_sim -> AdmittanceSim
#    4) durata e sussulti              5) run_viewer (finestra)  OPPURE  ciclo di sim.step() + analyse_and_plot
# =====================================================================================================
# COSA: argparse = lettura delle opzioni da riga di comando (--viewer, --fdes 8, ...).
import argparse
# COSA: csv = scrittura del file .csv (valori separati da virgole).  DOVE: analyse_and_plot, in fondo.
import csv
# COSA: time = orologio del computer (tempo REALE).  DOVE: run_viewer (tempo reale), ciclo batch (durata del calcolo).
import time
# COSA: Path per la cartella "out" e i nomi dei file salvati.
from pathlib import Path
import numpy as np

# COSA: panda_scene serve qui solo per la grafica (update_tissue_mesh, measure_tissue_compression).
import panda_scene as ps
# COSA: dal controllore: i parametri, la simulazione, le soglie (per plot_schedule) e il default di --goal-depth.
from admittance_control import AdmittanceParams, AdmittanceSim, StiffnessSchedule, GOAL_DEPTH
# COSA: generatore dei sussulti casuali (impulsi gaussiani, tissue.py).  DOVE: in fondo, --events-seed.
from tissue import gaussian_events
# COSA: le calibrazioni e i filtri disponibili sull'ATI: servono come elenco di scelte valide per --ati-cal/--ati-filter.
from ft_sensor import CALIBRATIONS, NETFT_FILTERS

# COSA: frequenza dei campioni nel log [Hz] = 1 / dt (dt = 1 ms, un campione per step()).
# DOVE: analyse_and_plot: controllo del passo, dF/dt in N/s, conteggio dei secondi (campioni / FS).
FS = 1000


def make_params(a):
    # COSA: parte dai default di AdmittanceParams (a = oggetto con le opzioni lette da argparse).
    p = AdmittanceParams()
    # COSA: coppie (nome dell'opzione, nome del campo). getattr(a, arg) legge l'opzione (es. a.fdes); se vale None
    #       l'utente non l'ha data e si tiene il default; altrimenti setattr(p, field, v) scrive p.F_DES = v.
    # PERCHÉ: un solo ciclo invece di 13 "if" quasi uguali.
    for arg, field in (("fdes", "F_DES"), ("M", "M_D"), ("kfar", "K_FAR"), ("knear", "K_NEAR"), ("dfar", "D_FAR"),
                       ("dnear", "D_NEAR"), ("doff", "D_OFF"), ("integrator", "INTEGRATOR"), ("kt_mode", "KT_MODE"),
                       ("scan_pattern", "SCAN_PATTERN"), ("row_spacing", "SCAN_ROW_SPACING"),
                       ("scan_speed", "SCAN_SPEED"), ("scan_delay", "SCAN_DELAY")):
        v = getattr(a, arg)
        if v is not None:
            setattr(p, field, v)
    # COSA: stesse cose per le opzioni a DUE numeri (nargs=2): argparse da' una lista [a, b] o None.
    #       "if v:" e' falso sia per None sia per lista vuota; tuple(v) la converte nel tipo dei default.
    for arg, field in (("scan_x", "SCAN_X_RANGE"), ("scan_y", "SCAN_Y_RANGE"),
                       ("scan_start", "SCAN_START"), ("scan_end", "SCAN_END")):
        v = getattr(a, arg)
        if v:
            setattr(p, field, tuple(v))
    # COSA: le opzioni-interruttore (store_true) valgono False se assenti, True se presenti.
    if a.b_include_kd:
        p.B_INCLUDE_KD = True
    if a.no_scan:
        p.SCAN_ENABLE = False
    # DOVE: build_sim (simulazione) e plot_schedule (grafico delle soglie).
    return p


def build_sim(a):
    # COSA: crea la simulazione con i parametri e le opzioni della discesa e del sensore.
    #       "None if a.ati_filter == 'none' else int(...)": l'opzione e' una stringa ("none", "35", ...), il sensore
    #       vuole None oppure un intero in Hz. Il seme del rumore non si passa: resta il default 0.
    # DOVE: in fondo: sim = build_sim(a). Qui dentro girano build_model, il caricamento della posa "hover" e la tara.
    sim = AdmittanceSim(make_params(a), descent_time=a.descent_time, goal_depth=a.goal_depth,
                        ati_calibration=a.ati_cal,
                        ati_filter=None if a.ati_filter == "none" else int(a.ati_filter))
    # COSA: stampa un riassunto del sensore: noise_std[2] = deviazione standard del rumore su Fz (6.25 mN -> "6.2"),
    #       range[2] = portata in Fz (35 N per SI-25-0.25).
    print(f"Sensore: ATI Nano17 {a.ati_cal}, filtro Net F/T {a.ati_filter}, "
          f"rumore (ipotesi) {sim.ft.noise_std[2]*1000:.1f} mN, portata Fz {sim.ft.range[2]:g} N")
    if sim.p.SCAN_ENABLE:
        # COSA: stima della durata = discesa (4 s) + attesa (2 s) + percorso (164 s) = 170 s. E' una stima: il
        #       contatto vero (F > 2 N) arriva a 3.6 s, non a 4.
        estimated = a.descent_time + sim.p.SCAN_DELAY + sim.traj.T
        print(f"Scansione {sim.p.SCAN_PATTERN}: {len(sim.traj.waypoints)} waypoint, "
              f"{sim.traj.length:.2f} m, circa {estimated:.0f} s con contatto continuo")
    return sim


def run_viewer(sim, duration, vis_gain=1.0):
    """Finestra MuJoCo interattiva: ~16 passi fisici (16 ms) per fotogramma, in tempo reale."""
    # COSA: import DENTRO la funzione: il viewer serve solo con --viewer (ha bisogno di uno schermo).
    import mujoco.viewer
    m, d, ids = sim.model, sim.data, sim.ids
    # COSA: apre la finestra in modo "passivo": il viewer NON fa avanzare la fisica, la facciamo avanzare noi
    #       con sim.step() e poi gli diciamo di ridisegnare (v.sync()). "with" chiude la finestra all'uscita.
    with mujoco.viewer.launch_passive(m, d) as v:
        # COSA: telecamera: angolo orizzontale 140 gradi, guarda dall'alto (-25 gradi), a 1.3 m, puntata sopra il
        #       centro del torso (0, 0, 0.15).
        v.cam.azimuth, v.cam.elevation, v.cam.distance = 140, -25, 1.3
        v.cam.lookat[:] = [0.0, 0.0, 0.15]
        # COSA: tempo simulato della prossima stampa a terminale.
        next_print = 0.0
        # COSA: continua finche' la finestra e' aperta e non si e' raggiunta la durata.
        while v.is_running() and d.time < duration:
            # COSA: orologio reale all'inizio del fotogramma.
            t_wall = time.time()
            # COSA: 16 passi fisici = 16 ms simulati per fotogramma (~60 fotogrammi al secondo).
            #       fn = forza VERA del tessuto restituita dall'ultimo step().
            for _ in range(16):
                fn = sim.step()
            # grafica: torso che respira, avvallamento sotto la sonda, colore della punta secondo la forza
            # COSA: quanto la sfera verde e' dentro la pelle (geometria sfera-piano, panda_scene).
            comp = ps.measure_tissue_compression(m, d, ids, sim.tissue, d.time)
            # COSA: centro della sfera nel mondo.
            centre = d.geom_xpos[ids["tip_geom"]]
            # COSA: ridisegna il torso; se comp > 0 passa (x, y, compressione) per l'avvallamento, altrimenti None.
            ps.update_tissue_mesh(m, sim.tissue, d.time, vis_gain, (centre[0], centre[1], comp) if comp > 0 else None)
            # COSA: due booleani in una riga: verde se la forza vera e' entro 1 N da F_DES, rosso se supera 15 N.
            green, red = abs(fn - sim.p.F_DES) <= 1.0, fn > 15.0
            # COSA: colore della sfera: verde, altrimenti rosso, altrimenti giallo (espressioni condizionali annidate).
            m.geom_rgba[ids["tip_geom"]] = (0.2, 0.8, 0.3, 1) if green else ((0.9, 0.2, 0.2, 1) if red else (0.95, 0.8, 0.2, 1))
            # COSA: rimanda l'heightfield modificato alla scheda grafica (senza questa riga il torso resterebbe fermo).
            v.update_hfield(ids["hfield"])
            # COSA: copia lo stato nel viewer e ridisegna; raccoglie anche le interazioni col mouse.
            v.sync()
            # COSA: ogni 0.5 s simulati stampa gli ultimi valori del log: forza vera, forza misurata, penetrazione
            #       (mm), K_d, K_t. L[...][-1] = ultimo elemento della lista.
            if d.time >= next_print:
                L = sim.log
                print(f"t={d.time:6.2f}s  F vera {fn:5.2f} N  F sensore {L['fmeas'][-1]:5.2f} N  "
                      f"penetrazione {L['delta'][-1] * 1000:5.2f} mm  K_d {L['Kd'][-1]:6.0f} N/m  K_t {L['Kt'][-1]:5.0f} N/m")
                next_print += 0.5
            # COSA: dorme quanto manca a 16 ms reali: 16 ms simulati durano 16 ms veri (tempo reale). Se il calcolo
            #       e' stato piu' lento, max(0, ...) = 0 e la simulazione va piu' lenta del vero.
            time.sleep(max(0.0, 0.016 - (time.time() - t_wall)))


def plot_schedule(p, out):
    # COSA: matplotlib con il backend "Agg" = disegna su file PNG senza aprire finestre. use() va chiamato PRIMA
    #       di importare pyplot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # COSA: le stesse soglie del controllore, con questi parametri.
    sch = StiffnessSchedule(p)
    # COSA: 600 distanze da -10 mm (dentro la pelle) a D_FAR + 30 mm = 90 mm.
    d = np.linspace(-0.010, p.D_FAR + 0.03, 600)
    # COSA: 3 grafici uno sotto l'altro con lo stesso asse x.
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    # COSA: grafico 1: K_d(d) con F_meas = 0 (cancello in forza aperto: si vede solo la parte di distanza).
    #       La list comprehension calcola sch.K per ognuna delle 600 distanze; d * 1000 = asse in mm.
    ax[0].plot(d * 1000, [sch.K(x, 0.0) for x in d]); ax[0].set_ylabel("K_d [N/m]")
    ax[0].set_title("Soglie di K_d: alta lontano, scende vicino, si spegne a contatto (senza forza misurata)")
    # COSA: grafico 2: forza desiderata attiva rho(d) * F_DES (0 lontano, F_DES sulla pelle).
    ax[1].plot(d * 1000, [sch.rho(x, 0.0) * p.F_DES for x in d], color="tab:green")
    ax[1].set_ylabel("forza di riferimento attiva [N]")
    # COSA: grafico 3: limite di velocita' verso la pelle, in mm/s.
    ax[2].plot(d * 1000, [sch.v_in_max(x, 0.0) * 1000 for x in d], color="tab:red")
    ax[2].set_ylabel("v massima verso la pelle [mm/s]"); ax[2].set_xlabel("distanza d dalla pelle [mm]  (>0 = sopra)")
    # COSA: su ogni grafico tre linee verticali tratteggiate alle soglie e la griglia.
    for a_ in ax:
        for x in (p.D_OFF, p.D_NEAR, p.D_FAR):
            a_.axvline(x * 1000, color="grey", ls=":", lw=1)
        a_.grid(alpha=0.3)
    # COSA: nomi delle soglie scritti in alto nel primo grafico (al 97% dell'altezza dell'asse y).
    for x, lab in ((p.D_OFF, "D_OFF"), (p.D_NEAR, "D_NEAR"), (p.D_FAR, "D_FAR")):
        ax[0].text(x * 1000, ax[0].get_ylim()[1] * 0.97, " " + lab, fontsize=8, va="top")
    # COSA: sistema i margini, salva il PNG a 120 dpi, chiude la figura (libera memoria).
    # DOVE: solo con --plot-schedule -> out/k_schedule.png. Non influenza la simulazione.
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close(fig)
    print("salvato", out)


def analyse_and_plot(sim, a, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = sim.p
    # COSA: converte ogni lista del log in array numpy (dict comprehension): cosi' si fanno conti su tutti i
    #       campioni insieme (medie, maschere booleane, ...).
    L = {k: np.asarray(v) for k, v in sim.log.items()}
    t = L["t"]
    # COSA: controlla che il passo medio fra campioni sia esattamente 1 ms (entro 1e-9).
    # PERCHÉ: tutte le statistiche "in secondi" dividono il numero di campioni per FS.
    assert abs(np.diff(t).mean() - 1.0 / FS) < 1e-9
    fm = L["fmeas"]
    # COSA: errore di forza rispetto a F_DES (NON a F_des_eff): durante la discesa vale 5 N, per questo le
    #       statistiche usano solo la finestra dopo il contatto (win).
    eF = p.F_DES - fm
    # COSA: maschera booleana: True dove la forza VERA supera 1 mN (log "ncontact").
    contact = L["ncontact"] > 0
    # COSA: se non c'e' mai stato contatto non ha senso fare statistiche: avvisa ed esce.
    if not contact.any():
        print("ATTENZIONE: nessun contatto nella simulazione. Controlla raggiungibilita' e durata.")
        return None
    # COSA: istante del primo contatto: il primo t dove la maschera e' True. Misurato: 3.333 s.
    t_c = t[contact][0]
    # COSA: finestra delle statistiche: da t_c + settle (3 s) alla fine -> 6.3-173.0 s. Esclude il transitorio.
    win = t >= t_c + a.settle
    # COSA: |dF/dt| della forza VERA nella finestra: differenze fra campioni consecutivi * FS = N/s.
    #       Indica i "picchi" (misurato: massimo 5.1 N/s).
    dfdt = np.abs(np.diff(L["fn"][win])) * FS
    # COSA: dizionario delle statistiche (misurato, simulazione di default):
    #   F_media/min/max     forza misurata nella finestra: 4.994 / 4.235 / 5.257 N
    #   picco_sopra_Fdes    quanto la forza misurata supera F_DES dopo il PRIMO contatto (anche fuori finestra): 0.257 N
    #   eF_media, eF_rms    errore medio (0.006) e quadratico medio sqrt(mean(eF^2)) (0.051 N)
    #   eF_max              errore piu' grande in modulo (0.765 N, durante i due sussulti sovrapposti a 23.6 s)
    #   entro_1N, entro_2N  percentuale di campioni con |eF| < 1 N e < 2 N (100 %)
    #   Kd_a_regime         K_d massimo nella finestra: 0 -> a contatto l'ammettenza e' davvero un integratore puro
    #   Bd_medio, Kt_min/max smorzamento medio (12.7) e rigidezze attraversate (500-1628 N/m)
    stats = dict(
        durata_s=t[-1], primo_contatto_s=t_c, finestra=f"{t_c + a.settle:.1f}-{t[-1]:.1f} s",
        F_media=fm[win].mean(), F_min=fm[win].min(), F_max=fm[win].max(),
        picco_sopra_Fdes=max(0.0, fm[t >= t_c].max() - p.F_DES),
        eF_media=eF[win].mean(), eF_rms=np.sqrt(np.mean(eF[win] ** 2)), eF_max=np.abs(eF[win]).max(),
        entro_1N=100 * np.mean(np.abs(eF[win]) < 1.0), entro_2N=100 * np.mean(np.abs(eF[win]) < 2.0),
        max_dFdt=dfdt.max(), Kd_a_regime=L["Kd"][win].max(), Bd_medio=L["Bd"][win].mean(),
        Kt_min=L["Kt"][win].min(), Kt_max=L["Kt"][win].max(),
    )
    # COSA: maschera "sto scansionando": all'inizio tutta False.
    scan = np.zeros_like(t, bool)
    if p.SCAN_ENABLE:
        # COSA: True dove l'avanzamento e' strettamente fra 0.1% e 99.9% E siamo nella finestra (& = "and"
        #       elemento per elemento): i campioni in cui la sonda si muove davvero sul percorso.
        scan = (L["scan_s"] > 0.001) & (L["scan_s"] < 0.999) & win
        if scan.any():
            # COSA: aggiunge al dizionario le statistiche della sola scansione (misurato):
            #   scansione_s 163.1 s; scan_eF_rms 0.052 N; scan_exy_medio/max 0.49 / 2.1 mm (errore XY);
            #   scan_completata_pct 100 (avanzamento finale); perdite_contatto_s 0 (secondi con F_meas < 1 N).
            stats.update(scansione_s=scan.sum() / FS, scan_eF_rms=np.sqrt(np.mean(eF[scan] ** 2)),
                         scan_entro_1N=100 * np.mean(np.abs(eF[scan]) < 1.0), scan_eF_max=np.abs(eF[scan]).max(),
                         scan_exy_medio_mm=1000 * L["exy"][scan].mean(), scan_exy_max_mm=1000 * L["exy"][scan].max(),
                         scan_completata_pct=100 * L["scan_s"][-1],
                         perdite_contatto_s=float(np.sum(L["fmeas"][win] < p.F_CONTACT_OK) / FS))
    # COSA: stampa intestazione e statistiche: i float con 3 decimali, il resto (la stringa "finestra") cosi' com'e'.
    #       {k:20s} = nome allineato su 20 caratteri. np.float64 conta come float per isinstance.
    print(f"\n=== ammettenza | M_d {p.M_D} kg | F_des {p.F_DES} N | {p.INTEGRATOR} | K_t {p.KT_MODE} ===")
    for k, v in stats.items():
        print(f"  {k:20s} {v:.3f}" if isinstance(v, float) else f"  {k:20s} {v}")

    # ------------------------------------------------------------------ grafico nel tempo
    # COSA: figura con 6 grafici uno sotto l'altro, stesso asse dei tempi.  DOVE: out/admittance_run.png.
    fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    # COSA: grafico 0 - forza vera (dal tessuto) e forza misurata (dal sensore), linea verde a F_DES e banda +-1 N.
    #       La differenza fra le due curve e' l'effetto del sensore (rumore, ritardo, inerzia della sonda).
    ax[0].plot(t, L["fn"], lw=0.8, alpha=0.6, label="forza vera")
    ax[0].plot(t, fm, lw=0.5, color="tab:orange", label="F misurata (sensore, 1 kHz)")
    ax[0].axhline(p.F_DES, color="g"); ax[0].axhspan(p.F_DES - 1, p.F_DES + 1, color="g", alpha=0.12)
    ax[0].set_ylabel("F normale [N]"); ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    # COSA: grafico 1 - errore di forza, linea dello zero, banda +-1 N e una scritta in alto a destra con
    #       RMS, % entro 1 N e picco. transform=transAxes: coordinate come frazione del riquadro (0.995, 0.93).
    ax[1].plot(t, eF, lw=0.5, color="tab:red"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axhspan(-1, 1, color="g", alpha=0.12); ax[1].set_ylabel("e_F = F_des - F_meas [N]")
    ax[1].text(0.995, 0.93, f"RMS {stats['eF_rms']:.2f} N | entro 1 N: {stats['entro_1N']:.0f}% | picco sopra F_des: "
               f"{stats['picco_sopra_Fdes']:.2f} N", transform=ax[1].transAxes, ha="right", va="top", fontsize=9)
    # COSA: grafico 2 - K_d (scala a sinistra); twinx() crea un secondo asse y a destra sullo stesso asse x per B_d.
    #       Si vede K_d che si spegne al contatto e B_d che segue la rigidezza del tessuto.
    ax[2].plot(t, L["Kd"], color="tab:blue"); ax[2].set_ylabel("K_d [N/m]", color="tab:blue")
    a2 = ax[2].twinx(); a2.plot(t, L["Bd"], color="tab:purple", lw=0.8); a2.set_ylabel("B_d [Ns/m]", color="tab:purple")
    # COSA: grafico 3 - velocita' dell'ammettenza e il suo limite (tratteggiato), in mm/s; a destra la distanza
    #       dalla pelle in mm (asse fra -40 e 100 mm).
    ax[3].plot(t, L["v_in"] * 1000, label="v verso la pelle", color="tab:green", lw=0.6)
    ax[3].plot(t, L["v_lim"] * 1000, ls="--", color="tab:red", label="limite")
    ax[3].set_ylabel("v [mm/s]"); ax[3].legend(loc="upper right", fontsize=8)
    a3 = ax[3].twinx(); a3.plot(t, L["dist"] * 1000, color="grey", lw=0.6)
    a3.set_ylabel("distanza dalla pelle [mm]", color="grey"); a3.set_ylim(-40, 100)
    # COSA: grafico 4 - quota della pelle sotto la sonda e quota della punta: la distanza fra le due e' la
    #       penetrazione; si vede la punta che segue respiro e sussulti.
    ax[4].plot(t, L["zs"] * 1000, label="superficie sotto la sonda", lw=0.8)
    ax[4].plot(t, L["ztip"] * 1000, label="punta sonda", lw=0.8)
    ax[4].set_ylabel("z [mm]"); ax[4].legend(loc="upper right", fontsize=8)
    # COSA: grafico 5 - x, y della punta (continue) e dei riferimenti della serpentina (tratteggiati); a destra K_t.
    ax[5].plot(t, L["x"] * 1000, label="x punta", lw=0.8); ax[5].plot(t, L["y"] * 1000, label="y punta", lw=0.8)
    ax[5].plot(t, L["x_ref"] * 1000, "--", lw=0.6, color="tab:blue"); ax[5].plot(t, L["y_ref"] * 1000, "--", lw=0.6, color="tab:orange")
    ax[5].set_ylabel("posizione [mm]"); ax[5].legend(loc="upper right", fontsize=8); ax[5].set_xlabel("t [s]")
    a5 = ax[5].twinx(); a5.plot(t, L["Kt"], color="tab:brown", lw=0.5); a5.set_ylabel("K_t del tessuto [N/m]", color="tab:brown")
    # COSA: su tutti i grafici griglia e banda grigia chiara sulla finestra delle statistiche.
    for x in ax:
        x.grid(alpha=0.3)
        x.axvspan(t_c + a.settle, t[-1], color="grey", alpha=0.05)
    # COSA: titolo generale con i parametri principali, poi salva e chiude.
    fig.suptitle(f"Ammettenza a integratore puro: F_des = {p.F_DES:g} N, M_d = {p.M_D:g} kg, {p.INTEGRATOR}, K_t {p.KT_MODE}")
    plt.tight_layout(); plt.savefig(outdir / "admittance_run.png", dpi=110); plt.close(fig)

    # ------------------------------------------------------------------ percorso e mappe sul torace
    # COSA: slice(None, None, 20) = "un elemento ogni 20" ([::20]): un punto ogni 20 ms.
    # PERCHÉ: 173 000 punti per grafico a dispersione sarebbero lenti e illeggibili; 8650 bastano.
    sub = slice(None, None, 20)                                  # un punto ogni 20 ms
    x_mm, y_mm = L["x"][sub] * 1000, L["y"][sub] * 1000
    # COSA: figura con 3 grafici affiancati, la pianta del torace vista dall'alto.  DOVE: out/admittance_scan_path.png.
    fig, bx = plt.subplots(1, 3, figsize=(17, 5.8))
    # COSA: grafico 0 - percorso vero della punta colorato col tempo (scatter con c = tempo) + percorso richiesto
    #       tratteggiato (qui con tutti i campioni, non sottocampionati).
    pts = bx[0].scatter(x_mm, y_mm, c=t[sub], s=3, cmap="viridis")
    bx[0].plot(L["x_ref"] * 1000, L["y_ref"] * 1000, "--", color="tab:orange", lw=0.8, label="percorso richiesto")
    bx[0].set_title("Percorso della sonda"); fig.colorbar(pts, ax=bx[0], label="tempo [s]"); bx[0].legend(fontsize=8)
    # COSA: maschera della finestra sottocampionata allo stesso modo dei punti.
    m2 = win[sub]
    # COSA: grafico 1 - |errore di forza| punto per punto sul torace. "magma_r" = scala invertita (scuro = errore
    #       grande); scala fissa da 0 a max(1.5, 1.5 * RMS) = 1.5 N, cosi' errori piccoli restano chiari.
    pts = bx[1].scatter(x_mm[m2], y_mm[m2], c=np.abs(eF[sub][m2]), s=3, cmap="magma_r", vmin=0, vmax=max(1.5, 1.5 * stats["eF_rms"]))
    bx[1].set_title("Errore di forza |F_des - F_meas| sul torace"); fig.colorbar(pts, ax=bx[1], label="[N]")
    # COSA: grafico 2 - rigidezza K_t sotto la sonda: e' la StiffnessMap "vista" lungo il percorso (si vedono
    #       torace/addome, costole e sterno).
    pts = bx[2].scatter(x_mm[m2], y_mm[m2], c=L["Kt"][sub][m2], s=3, cmap="viridis")
    bx[2].set_title("Rigidita' del tessuto K_t sotto la sonda"); fig.colorbar(pts, ax=bx[2], label="[N/m]")
    # COSA: etichette, stessa scala in x e y (set_aspect "equal": 1 mm in x = 1 mm in y, niente deformazione), griglia.
    for b in bx:
        b.set(xlabel="x cranio-caudale [mm]", ylabel="y trasversale [mm]"); b.set_aspect("equal", adjustable="box"); b.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "admittance_scan_path.png", dpi=120); plt.close(fig)

    # ------------------------------------------------------------------ CSV
    # COSA: passo di scrittura: almeno 1 (stride 10 = un campione ogni 10 ms = 100 Hz).
    stride = max(1, a.csv_stride)
    # COSA: apre out/admittance_run.csv in scrittura ("w" lo sovrascrive). newline="" e' quello che chiede il modulo
    #       csv (evita righe vuote in piu' su Windows). "with" chiude il file alla fine.
    with (outdir / "admittance_run.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        # COSA: riga d'intestazione con i nomi delle colonne (unita' nel nome: _N, _mm, _mm_s).
        w.writerow(["t_s", "F_true_N", "F_meas_N", "F_des_N", "e_F_N", "K_d", "B_d", "K_t", "v_in_mm_s", "dist_mm",
                    "z_surface_mm", "z_tip_mm", "x_mm", "y_mm", "x_ref_mm", "y_ref_mm", "scan_progress"])
        # COSA: un campione ogni stride: ogni valore formattato come testo con decimali fissi (:.4f = 4 decimali),
        #       metri convertiti in mm (* 1000).
        for i in range(0, len(t), stride):
            w.writerow([f"{t[i]:.3f}", f"{L['fn'][i]:.4f}", f"{fm[i]:.4f}", p.F_DES, f"{eF[i]:.4f}", f"{L['Kd'][i]:.2f}",
                        f"{L['Bd'][i]:.2f}", f"{L['Kt'][i]:.1f}", f"{L['v_in'][i] * 1000:.3f}", f"{L['dist'][i] * 1000:.3f}",
                        f"{L['zs'][i] * 1000:.3f}", f"{L['ztip'][i] * 1000:.3f}", f"{L['x'][i] * 1000:.2f}", f"{L['y'][i] * 1000:.2f}",
                        f"{L['x_ref'][i] * 1000:.2f}", f"{L['y_ref'][i] * 1000:.2f}", f"{L['scan_s'][i]:.5f}"])
    print(f"salvati {outdir}/admittance_run.png, admittance_scan_path.png e admittance_run.csv (1 campione ogni {stride} ms)")
    return stats


# COSA: questo blocco gira solo se il file viene ESEGUITO (python run_admittance.py), non se viene importato.
if __name__ == "__main__":
    # COSA: il lettore delle opzioni. Ogni add_argument dichiara un'opzione: nome, tipo, default, scelte valide.
    #       argparse converte i trattini in trattini bassi: --kt-mode -> a.kt_mode, --events-seed -> a.events_seed.
    ap = argparse.ArgumentParser()
    # COSA: durata simulata; None = calcolata dopo (in fondo).
    ap.add_argument("--duration", type=float, default=None, help="default: automatica (discesa + attesa + percorso + 3 s)")
    # COSA: store_true = interruttore: False se assente, True se scritto.
    ap.add_argument("--viewer", action="store_true")
    # COSA: esagerazione del respiro nella grafica (update_tissue_mesh).
    ap.add_argument("--vis-gain", type=float, default=1.0)
    ap.add_argument("--plot-schedule", action="store_true")
    # COSA: calibrazione ATI fra le chiavi di CALIBRATIONS (sorted = in ordine alfabetico).
    ap.add_argument("--ati-cal", choices=sorted(CALIBRATIONS), default="SI-25-0.25", help="calibrazione Nano17")
    # COSA: filtro: "none" piu' le frequenze di NETFT_FILTERS come stringhe ("if f" scarta il None della lista).
    ap.add_argument("--ati-filter", choices=["none"] + [str(f) for f in NETFT_FILTERS if f], default="none",
                    help="filtro passa-basso del Net F/T [Hz]")
    # COSA: seme dei sussulti (stesso seme = stessi sussulti); -1 = nessuno.
    ap.add_argument("--events-seed", type=int, default=0, help="seme dei sussulti gaussiani casuali; -1 = nessun sussulto")
    # COSA: durata della discesa del planner e profondita' dell'obiettivo (vanno in AdmittanceSim).
    ap.add_argument("--descent-time", type=float, default=4.0)
    ap.add_argument("--goal-depth", type=float, default=GOAL_DEPTH, help="profondita' dell'obiettivo del planner [m]")
    # COSA: secondi dopo il primo contatto esclusi dalle statistiche; passo del CSV.
    ap.add_argument("--settle", type=float, default=3.0, help="secondi dopo il primo contatto prima delle statistiche")
    ap.add_argument("--csv-stride", type=int, default=1, help="un campione CSV ogni N passi (1 = 1 kHz)")
    # COSA: 7 opzioni numeriche dichiarate in un ciclo (f"--{name}"); default None = "usa il default di
    #       AdmittanceParams" (vedi make_params).
    for name in ("fdes", "M", "kfar", "knear", "dfar", "dnear", "doff"):
        ap.add_argument(f"--{name}", type=float, default=None)
    # COSA: scelte del controllore e della scansione (None = default di AdmittanceParams).
    ap.add_argument("--integrator", choices=("explicit", "implicit"), default=None)
    ap.add_argument("--kt-mode", choices=("k_eq", "tangent"), default=None)
    ap.add_argument("--b-include-kd", action="store_true")
    ap.add_argument("--no-scan", action="store_true", help="niente traiettoria XY")
    ap.add_argument("--scan-pattern", choices=("serpentine", "line"), default=None)
    # COSA: nargs=2 = l'opzione vuole DUE numeri (es. --scan-x -0.16 0.16); metavar = come compaiono nell'aiuto.
    ap.add_argument("--scan-x", type=float, nargs=2, metavar=("XMIN", "XMAX"), help="limiti cranio-caudali della serpentina [m]")
    ap.add_argument("--scan-y", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="limiti trasversali della serpentina [m]")
    ap.add_argument("--row-spacing", type=float, default=None, help="distanza massima tra le passate [m]")
    ap.add_argument("--scan-start", type=float, nargs=2, metavar=("X", "Y"), help="inizio della linea singola")
    ap.add_argument("--scan-end", type=float, nargs=2, metavar=("X", "Y"), help="fine della linea singola")
    ap.add_argument("--scan-speed", type=float, default=None, help="velocita' media lungo il percorso [m/s]")
    ap.add_argument("--scan-delay", type=float, default=None, help="attesa dopo il primo contatto [s]")
    # COSA: legge davvero la riga di comando e restituisce l'oggetto a con un attributo per opzione.
    a = ap.parse_args()

    # COSA: cartella "out" RELATIVA alla cartella da cui lanci il comando (lanciare da project_1.1);
    #       exist_ok=True: nessun errore se esiste gia'.
    outdir = Path("out"); outdir.mkdir(exist_ok=True)
    # COSA: con --plot-schedule disegna prima il grafico delle soglie, poi la simulazione parte comunque.
    if a.plot_schedule:
        plot_schedule(make_params(a), outdir / "k_schedule.png")
    # COSA: costruisce la simulazione (modello, posa di partenza, tara) e stampa il riassunto.
    sim = build_sim(a)
    # COSA: durata automatica: con la serpentina discesa + attesa + percorso + 3 s di margine = 4 + 2 + 164 + 3
    #       = 173 s; senza scansione 15 s.
    if a.duration is None:
        a.duration = (a.descent_time + sim.p.SCAN_DELAY + sim.traj.T + 3.0) if sim.p.SCAN_ENABLE else 15.0
    print(f"durata simulazione: {a.duration:.1f} s")
    if a.events_seed >= 0:
        # COSA: genera i sussulti fino alla fine della simulazione e li mette nel tessuto che la simulazione usa
        #       (sostituisce la lista vuota). Da qui Tissue.surface_height li somma alla pelle.
        #       Misurato con seme 0: 23.6 s (5.2 mm), 23.6 s (9.6 mm), 37.1 s (7.7 mm), 158.3 s (5.0 mm).
        # PERCHÉ dopo build_sim: la tara e' fatta a t = 0, e comunque il primo sussulto non arriva prima di 10 s.
        sim.tissue.events = gaussian_events(seed=a.events_seed, t_end=a.duration)
        # COSA: stampa l'elenco: join unisce le stringhe con ", " (t0, ampiezza in mm, sigma).
        print("sussulti gaussiani: " + ", ".join(f"{e.t0:.1f} s ({e.amplitude*1000:.1f} mm, sigma {e.duration:.2f} s)"
                                                  for e in sim.tissue.events))
    # COSA: due modi di girare. Col viewer: finestra in tempo reale e NIENTE grafici/CSV alla fine.
    if a.viewer:
        run_viewer(sim, a.duration, a.vis_gain)
    else:
        # COSA: modo batch: chiama step() (1 ms ciascuno) finche' il tempo simulato raggiunge la durata, il piu'
        #       in fretta possibile (misurato: 173 s simulati in 36 s reali), poi statistiche, grafici e CSV.
        t0 = time.time()
        while sim.data.time < a.duration:
            sim.step()
        print(f"simulati {a.duration:.0f} s in {time.time() - t0:.0f} s di tempo reale")
        analyse_and_plot(sim, a, outdir)
