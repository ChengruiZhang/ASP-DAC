Codes of paper "Thermal-Aware Layout Optimization and Mapping Methods for Resistive Neuromorphic Engines"
https://ieeexplore.ieee.org/abstract/document/9712596?casa_token=abGP_OpH3gEAAAAA:vaX3peKqeejW31ZLFaB3hpsuLFYluggT9KDkaae68Ob3zwBr3FTbvWotMbh3jiuYugLdXM5kLg

The files snn-node05.py ann-node05.py and ./self_modes come from https://github.com/nitin-rathi/hybrid-snn-conversion and we change these files to adapt our methods.

The file ./HotSpot is the chip-level thermal simulator (https://github.com/uvahotspot/hotspot). We change some of the C++ code to adapt our demands and our floorplanning files are stored in this directory.

The other files implement our re-ordering methods and the corresponding comparison with other method. The main func is remap_final.py.