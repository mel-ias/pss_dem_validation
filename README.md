# p4s_change_detection
Runs change detection and statistics using demcoreg


Privates Repo um Veränderungserkennung und Genauigkeitsanalysen mittels DEMs durchzuführen. 
Verwendet DEMCOREG von Shean et al. (2016), Install-Anleitung:

Install pygeotools and demcoreg from latest github source:
```
python -m pip install git+https://github.com/dshean/pygeotools.git
python -m pip install git+https://github.com/dshean/demcoreg.git
python -m pip install git+https://github.com/dshean/imview.git
Requires some reworking of PATH and ability to call scripts.
 ```
Benötigt Funktionen aus dem Conda Environment von planet4stereo (installiere demcoreg wie o.g. in das Environment rein wenn nicht schon geschehen). 

#Todo: RPCM Funktion aus planet4stereo hierein schieben
