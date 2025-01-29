# p4s_change_detection
Small toolchain to validate results from Planet4Stereo (https://github.com/mel-ias/planet4stereo) using demcoreg (Shean et al., 2016)

### How to setup? 

1. setup planet4stereo and activate the conda environment 
2. install pygeotools and demcoreg from latest github source:
```
python -m pip install git+https://github.com/dshean/pygeotools.git
python -m pip install git+https://github.com/dshean/demcoreg.git
python -m pip install git+https://github.com/dshean/imview.git
 ```

3. call validation.py with arguments as specified in the respective file