# Git e GitHub --- guida pratica per il progetto di tirocinio

Questa guida raccoglie i comandi Git che userai più spesso da VS Code. I
comandi si eseguono nel terminale aperto **nella cartella principale del
repository**.

------------------------------------------------------------------------

## 1. Il modello mentale: dove finiscono le modifiche?

-   **Working directory**: i file che stai modificando sul PC.
-   **Staging area**: le modifiche che hai selezionato per il prossimo
    commit.
-   **Repository locale**: la cronologia dei commit salvata sul PC.
-   **Repository remoto (GitHub)**: la copia online del repository.

Il flusso tipico è:

``` bash
git status
git add nome_file.py       # oppure: git add .
git commit -m "Descrizione chiara"
git push
```

`commit` salva una versione nella cronologia locale; `push` invia i
commit a GitHub. Non sono la stessa cosa.

------------------------------------------------------------------------

## 2. Controlli quotidiani

``` bash
git status
```

Mostra il branch attuale e i file modificati, non tracciati o già messi
in staging.

``` bash
git diff
```

Mostra le modifiche non ancora messe in staging.

``` bash
git diff --staged
```

Mostra le modifiche già messe in staging, pronte per il commit.

``` bash
git log --oneline --graph --decorate --all
```

Mostra la cronologia dei commit in forma compatta.

``` bash
git branch
```

Elenca i branch locali; l'asterisco indica quello attivo.

``` bash
git remote -v
```

Mostra gli indirizzi dei repository remoti configurati.

------------------------------------------------------------------------

## 3. Salvare il lavoro: add, commit e push

### Selezionare le modifiche

``` bash
git add nome_file.py
```

Aggiunge un singolo file allo staging.

``` bash
git add .
```

Aggiunge tutte le modifiche nella cartella corrente e nelle sue
sottocartelle. Prima controlla `git status`, per evitare di includere
file indesiderati.

Per togliere un file dallo staging senza perdere le modifiche:

``` bash
git restore --staged nome_file.py
```

### Creare un commit

``` bash
git commit -m "Add force sensor low-pass filter"
```

Un commit dovrebbe rappresentare un cambiamento coerente e
comprensibile. Meglio diversi commit piccoli e descrittivi che un unico
commit enorme.

Esempi di messaggi: - `Add force sensor model` -
`Implement admittance controller` - `Fix velocity estimation` -
`Add contact detection test` - `Update controller documentation`

### Caricare su GitHub

``` bash
git push
```

Invia a GitHub i commit del branch corrente.

Al primo push di un nuovo branch:

``` bash
git push -u origin nome-branch
```

L'opzione `-u` collega il branch locale al corrispondente branch remoto;
in seguito, normalmente, basterà `git push`.

------------------------------------------------------------------------

## 4. Branch: sviluppare senza rischiare il ramo principale

Un **branch** è una linea di sviluppo separata. Una convenzione
semplice:

-   `main`: versione principale, mantenuta stabile.
-   `feature/...`: nuove funzionalità.
-   `fix/...`: correzioni di bug.
-   `experiment/...`: prove e sperimentazioni.

### Creare un branch e passarci

``` bash
git switch -c feature/force-filter
```

Crea il branch e lo rende attivo.

### Cambiare branch

``` bash
git switch main
git switch feature/force-filter
```

Prima di cambiare branch, controlla `git status`: non lasciare modifiche
non salvate che potrebbero complicare il passaggio.

### Elencare i branch

``` bash
git branch
git branch -a
```

Il secondo mostra anche i branch remoti conosciuti localmente.

### Rinominare il branch attuale

``` bash
git branch -m nuovo-nome
```

### Eliminare un branch locale già integrato

``` bash
git branch -d nome-branch
```

`-d` protegge da alcune eliminazioni accidentali di branch non ancora
integrati. `-D` forza l'eliminazione: usalo solo se sei certo di non
aver bisogno del lavoro contenuto lì.

------------------------------------------------------------------------

## 5. Integrare un branch in main

Quando la funzionalità è pronta e verificata:

``` bash
git switch main
git pull
git merge feature/force-filter
```

-   `switch main`: torna sul ramo principale.
-   `pull`: recupera eventuali aggiornamenti da GitHub.
-   `merge`: integra nella branch corrente il lavoro del branch
    indicato.

Se tutto è corretto, carica il risultato:

``` bash
git push
```

Poi puoi eliminare il branch locale:

``` bash
git branch -d feature/force-filter
```

### Alternativa consigliata quando collabori: Pull Request

1.  Esegui il push del branch:

    ``` bash
    git push -u origin feature/force-filter
    ```

2.  Apri GitHub e crea una **Pull Request** da `feature/force-filter`
    verso `main`.

3.  Controlla le differenze, descrivi cosa hai fatto e chiedi una
    revisione se lavori con un tutor o un collega.

4.  Dopo l'approvazione, completa il merge su GitHub.

La Pull Request permette di revisionare il codice prima di integrarlo.

------------------------------------------------------------------------

## 6. Sincronizzarsi con GitHub

``` bash
git fetch
```

Scarica le informazioni sui cambiamenti remoti, senza integrarli
automaticamente nel branch corrente.

``` bash
git pull
```

Recupera e integra gli aggiornamenti del branch remoto collegato a
quello locale. Prima di iniziare una sessione, può essere utile se
lavori anche da altri computer o con collaboratori.

``` bash
git push
```

Invia i tuoi commit al repository remoto.

**Regola pratica:** prima di iniziare a lavorare, se il repository è
condiviso o usato da più computer, esegui `git pull`. A fine sessione,
dopo aver creato i commit, esegui `git push`.

------------------------------------------------------------------------

## 7. Annullare o recuperare modifiche

Questi comandi sono utili, ma alcuni sono distruttivi: controlla sempre
`git status` prima di usarli.

### Togliere un file dallo staging

``` bash
git restore --staged nome_file.py
```

Non elimina le modifiche al file: lo rimuove soltanto dallo staging.

### Scartare le modifiche non ancora committate di un file

``` bash
git restore nome_file.py
```

**Attenzione:** elimina le modifiche locali non committate a quel file.

### Correggere l'ultimo messaggio di commit

``` bash
git commit --amend -m "Messaggio corretto"
```

Modifica l'ultimo commit. Evita di farlo su commit già condivisi con
altri, a meno che il gruppo non abbia concordato come procedere.

### Annullare un commit già condiviso in modo tracciabile

``` bash
git revert HASH_DEL_COMMIT
```

Crea un nuovo commit che annulla le modifiche di quello indicato. È
spesso più sicuro di riscrivere la cronologia condivisa.

### Tornare a un commit precedente

Per consultare la cronologia:

``` bash
git log --oneline
```

Per annullare in modo sicuro un commit già pubblicato, preferisci
`git revert`. Evita `git reset --hard` se non conosci bene le
conseguenze: può eliminare modifiche locali e spostare la cronologia.

------------------------------------------------------------------------

## 8. Se compare un conflitto di merge

Un conflitto significa che Git non può combinare automaticamente alcune
modifiche.

1.  Esegui `git status` per vedere i file in conflitto.
2.  Apri ciascun file segnalato in VS Code.
3.  Cerca i marcatori `<<<<<<<`, `=======` e `>>>>>>>`.
4.  Scegli o combina le modifiche corrette, rimuovendo i marcatori.
5.  Verifica il codice e i test.
6.  Segna il file come risolto e completa il merge:

``` bash
git add nome_file.py
git commit
```

Se il conflitto è avvenuto durante un merge e vuoi annullare il
tentativo:

``` bash
git merge --abort
```

Non scegliere automaticamente una delle due versioni: verifica quale
codice deve rimanere.

------------------------------------------------------------------------

## 9. Il file `.gitignore`

Il file `.gitignore`, nella cartella principale, indica a Git quali file
non deve iniziare a tracciare. Per un progetto Python puoi partire da:

``` gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# Jupyter
.ipynb_checkpoints/

# Log e temporanei
*.log
*.tmp

# Credenziali e variabili d'ambiente
.env

# Sistema operativo
.DS_Store
Thumbs.db
```

Non ignorare automaticamente cartelle o file necessari a riprodurre gli
esperimenti. Per esempio, valuta come versionare configurazioni,
dataset, log e risultati sperimentali in base alle regole del
laboratorio.

**Importante:** `.gitignore` non rimuove dal repository un file che Git
sta già tracciando. Inoltre non cancella un segreto già caricato: se hai
pubblicato una password o un token, revocalo/cambialo subito e concorda
come ripulire la cronologia.

------------------------------------------------------------------------

## 10. Organizzazione consigliata per un progetto di robotica

Una possibile struttura, da adattare al progetto reale:

``` text
robotic-ultrasound-control/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── main.py
│   ├── admittance_control.py
│   ├── force_sensor.py
│   └── robot_interface.py
├── config/
│   └── controller_params.yaml
├── tests/
├── experiments/
│   └── README.md
└── docs/
    └── architecture.md
```

Il `README.md` dovrebbe spiegare obiettivo, architettura, dipendenze,
installazione ed esecuzione. Mantieni separati codice, configurazioni,
test, documentazione e dati sperimentali.

Prima di caricare codice, dati o configurazioni del laboratorio su
GitHub, verifica le regole del tirocinio e del tutor. Usa un repository
privato se richiesto; non caricare credenziali, dati sensibili o
materiale non autorizzato.

------------------------------------------------------------------------

## 11. Flusso di lavoro consigliato

### Inizio sessione

``` bash
git status
git switch main
git pull
```

### Inizio di una nuova funzionalità

``` bash
git switch -c feature/nome-funzionalita
```

### Durante il lavoro

``` bash
git status
git diff
```

### Quando completi un'unità di lavoro

``` bash
git add nome_file.py
git diff --staged
git commit -m "Descrivi la modifica"
```

### Fine sessione

``` bash
git push -u origin feature/nome-funzionalita
```

Se il branch era già collegato al remoto, basta `git push`.

### Quando la funzionalità è verificata

Apri una Pull Request verso `main`, oppure integra il branch con
`git merge` secondo il flusso concordato con il tuo tutor.

------------------------------------------------------------------------

## 12. Mini glossario

  Termine             Significato
  ------------------- -----------------------------------------------------------
  Repository / repo   Cartella di progetto gestita da Git
  Commit              Versione registrata nella cronologia
  Staging             Modifiche selezionate per il prossimo commit
  Branch              Linea di sviluppo separata
  Merge               Integrazione di un branch in un altro
  Remote              Repository ospitato altrove, ad esempio su GitHub
  Push                Invio dei commit al repository remoto
  Pull                Recupero e integrazione degli aggiornamenti remoti
  Fetch               Recupero delle informazioni remote senza integrarle
  Clone               Copia locale di un repository remoto
  Pull Request        Proposta di integrazione e revisione delle modifiche
  Conflict            Modifiche incompatibili che richiedono una scelta manuale

------------------------------------------------------------------------

## 13. I comandi da ricordare a memoria

``` bash
git status
git pull
git switch -c feature/nome
git add .
git diff --staged
git commit -m "Descrizione"
git push -u origin feature/nome
git switch main
git merge feature/nome
git log --oneline --graph --decorate --all
```

**Abitudine migliore:** controlla `git status` prima e dopo le
operazioni importanti. Fai commit piccoli, con messaggi chiari, e fai
push regolarmente. GitHub è una copia remota utile, ma non sostituisce
backup e procedure di conservazione dei dati del laboratorio.
