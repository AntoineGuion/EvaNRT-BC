# EvaNRT-BC

------------------
Program objective: 
- Code of "EvaNRT-BC" that aims to evaluate hourly EC concentrations (total, residential and traffic contribution) from CAMS analyses using AE33 observation data in near real time for a chosen day. In addition, the daily evaluation is transposed into the Model Quality Indicator framework proposed in the FAIRMODE Guidance Document on Modelling Quality Objectives and Benchmarking.
------------------

------------------
Program structure:
- Launched with Main_workflow.sh composed of 4 python scripts; respectively for (1) the import of CAMS simulation (personal API key to be provided), (2) the import of NRT AE33 measurements from sftp server (login and password for server access to be provided), (3) the preparation of data and (4) the evaluation of simulations at station points including several diagnostics.

- The "inputs" folder should contain the prepared data (after script 3), "material" the station coordinates, "scripts" all the above-mentioned scripts and "outputs" the produced figures.
------------------

------------------
How to cite :
Guion, A., Gherras, M., Favez, O., & Colette, A. (2025). Demonstrator "EvaNRT-BC" for an NRT evaluation of the modeled Elemental / Black Carbon. Zenodo. https://doi.org/10.5281/zenodo.16812466
------------------
------------------
Authors and contact: 
- Antoine Guion (antoine.guion@ineris.fr), Mohamed Gherras, Olivier Favez, Augustin Colette 
------------------

------------------
Author's affiliation: 
- French National Institute for Industrial Environment and Risks (INERIS), Verneuil-en-Halatte, France
------------------

------------------
Related project: 
- https://riurbans.eu/ (funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101036245)
------------------

------------------
Extra documentation: 
- https://riurbans.eu/wp-content/uploads/2024/04/RI-URBANS_D19_D3_4.pdf (Deliverable D19 (D3.4) High resolution mapping over European urban areas) 
------------------
