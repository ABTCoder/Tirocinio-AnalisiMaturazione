# Package
I package utilizzati nel *virtual environment* sono i seguenti:

 - Matplotlib
 - Numpy
 - Scipy
 - OpenCV
 - SciKit-image
 - SciKit-learn
 - MATLAB Engine for Python

Utilizzare le versioni più recenti.
Altri package verranno automaticamente scaricati durante l'installazione dei precedenti.

## Installazione ed interfacciamento di MATLAB
Una volta installata l'ultima versione di MATLAB seguire le istruzioni in questa pagina per l'installazione dell'engine [Install MATLAB Engine API for Python](https://it.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
Tuttavia se si desidera installarlo nell'ambiente virtuale del progetto *Pycharm* bisogna sostituire temporaneamente il percorso dell'eseguibile di *Python3* nelle variabili d'ambiente con quello presente in **venv/Scripts**, infine seguire gli stessi passi del link sopra citato. In questo modo il MATLAB engine verrà installato nel progetto locale. Al termine non dimenticare di rimettere il percorso originale di Python nelle variabili d'ambiente.
Per utilizzare lo script di rilevamento d'ellissi in Python bisogna prima installare una versione di Visual Studio compatibile con la versione di MATLAB utilizzata per sfruttare il compilatore C++ incluso. Fatto ciò utilizzare il seguente comando in MATLAB:
`mex -setup cpp`
e selezionare il compilatore di Visual Studio.
Dopo di che compilare il file `generateEllipseCandidates.cpp` in MATLAB seguendo le istruzioni nella pagina Github del codice [https://github.com/AlanLuSun/High-quality-ellipse-detection](https://github.com/AlanLuSun/High-quality-ellipse-detection).

# Generazione del dataset
Se si desidera ricreare uno dei due dataset una volta modificati i parametri delle varie funzioni di rilevamento o miglioramento eseguire il seguente codice in **main<span>.py</span>**:

	successes = 0  
	total = 0  
	for k in range(70):  
	    _, s, t = extract_histograms("images/{0}.jpg".format(k+1), "labels/{0}.txt".format(k+1), min_mask=20, writelog=True, writedataset=True)  
	    successes = successes + s  
	    total = total + t  
  
	percent = (successes / total) * 100  
	result = open("result.txt", 'a')  
	result.write(str(successes)+" SU "+str(total)+" SUCCESSI, {:.2f}".format(percent)+"%\n")  
	result.close()
Utilizzare `"images/{0}.jpg".format(k+1)` e `"labels/{0}.txt".format(k+1)` per il primo dataset.
Utilizzare `"images/olives{0}.jpg".format(k)` e `"labels/olives{0}.txt".format(k)` per il secondo dataset.
Spostare i file generati nelle rispettive cartelle dei dataset.

Se si vogliono anche ridefinire i gradi di maturazione eseguire questo codice:

	for k in range(70):
		utils.write_ripening_csv("images/{0}.jpg".format(k+1), "labels/{0}.txt".format(k+1))
	utils.create_masked_ripening():
E nuovamente spostare i file generati (*maturazioni.csv*, *masked_ripening.txt* e *log2.txt*) nella rispettiva cartella.

# TEST DEI CLASSIFICATORI
Per testare i classificatori eseguire la funzione `calc_f1_score(dataset)` in  **main<span>.py</span>** dove `dataset` può essere 1, 2 o *'both'* per unire entrambi i dataset. Verranno salvato i punteggi nella cartella *scores* e le matrici di confusione in *confusion_matrixes*.
