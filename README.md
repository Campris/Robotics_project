# Controllo di ammettenza per ecografia robotica (simulazione MuJoCo)

Franka Panda controllato in coppia, con una sonda a punta sferica e un sensore forza/coppia ATI Nano17 alla flangia.
Il robot scende su un torso che respira, mantiene una forza di contatto di 5 N e scansiona la superficie a serpentina.

Legge di controllo: ammettenza a integratore puro → cinematica inversa differenziale → PD di giunto in coppia.

```
M_d · dv/dt + B_d · v = F_virt + (F_des − F_meas)        B_d = 2 · sqrt(M_d · K_t)
```

## Installazione

Serve Python 3.10 o successivo.

```
pip install -r requirements.txt
```

Il modello del Panda è già nella cartella `franka_emika_panda/`: non serve scaricare altro.

## Esecuzione

Lanciare i comandi dalla cartella del repository.

```
python run_admittance.py --viewer                    # finestra MuJoCo in tempo reale
python run_admittance.py                             # simulazione completa (173 s), grafici e CSV in out/
python run_admittance.py --no-scan --duration 12     # solo discesa e mantenimento della forza
python run_admittance.py --plot-schedule --no-scan   # anche il grafico di K_d in funzione della distanza
```

Senza `--viewer` il programma stampa le statistiche della forza e salva in `out/` due grafici e un CSV.
L'elenco completo delle opzioni è in testa a `run_admittance.py` (oppure `python run_admittance.py --help`).

## File

| File | Contenuto |
|---|---|
| `run_admittance.py` | Punto di ingresso: opzioni, viewer, statistiche, grafici e CSV |
| `admittance_control.py` | Controllore: soglie di rigidezza, ammettenza, traiettoria, IK differenziale, PD di giunto |
| `panda_scene.py` | Scena MuJoCo: Panda, sonda, sensore F/T, grafica del torso |
| `tissue.py` | Tessuto: geometria del torso, respiro, sussulti, contatto Hunt-Crossley |
| `ft_sensor.py` | Modello del sensore ATI Nano17 con box Net F/T |
| `franka_emika_panda/` | Modello MuJoCo del Panda senza mano |

Il contatto non è calcolato da MuJoCo: la forza del tessuto viene dal modello Hunt-Crossley in `tissue.py` ed è
applicata alla sonda come forza esterna, così la misura anche il sensore alla flangia.

## Sviluppo previsto

Lo smorzamento `B_d` usa oggi la rigidezza vera del tessuto nel punto di contatto, nota solo in simulazione.
Nella fase successiva sarà calcolato una sola volta da una stima unica della rigidezza, come sul robot reale.
Un agente SAC lo modulerà di ±20 % e sommerà una forza correttiva all'ingresso dell'ammettenza.


