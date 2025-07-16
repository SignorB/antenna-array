# Beamforming con Array Non Uniforme Ottimizzato

Questo progetto MATLAB calcola e confronta l’**array factor** di antenne con disposizione **non uniforme ottimizzata** rispetto a un array uniforme.

## Esecuzione

Per eseguire il progetto, è sufficiente lanciare lo script principale (`main.m`).

## Parametri Modificabili

- `theta_fixed`: direzione di puntamento desiderata (in radianti);
- `theta_new`: angolo a cui si presenza il picco a seguito dello steering
- `gradi_maschera`: ampiezza angolare su cui ottimizzare;
- `lambda`: lunghezza d’onda;
- `delta_opt_picco` / `delta_opt_sll`: vettori di fasi già precalcolate.

## Output

Lo script genera grafici dell’array factor in forma lineare e in dB, sia in scala cartesiana che polare, mostrando il confronto tra array ottimizzato e uniforme.
