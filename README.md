# **p4s_change_detection**  

A lightweight toolchain for validating results from [Planet4Stereo](https://github.com/mel-ias/planet4stereo) using [demcoreg](https://github.com/dshean/demcoreg) (*Shean et al., 2016*).  

## **Setup Instructions**  

### **1. Set up the Planet4Stereo Conda Environment**  
Ensure that the Planet4Stereo Conda environment is correctly installed and activated before proceeding.  

### **2. Install Required Dependencies**  
Install `pygeotools`, `demcoreg`, and `imview` from their latest GitHub sources:  

```
python -m pip install git+https://github.com/dshean/pygeotools.git
python -m pip install git+https://github.com/dshean/demcoreg.git
python -m pip install git+https://github.com/dshean/imview.git
```  

### **3. Run the Validation Script**  
Execute `validation.py` with the appropriate arguments as specified in the argument section of the script. To view available parameters and their descriptions, run:  

```
python validation.py -h
```  