# Transfer Learning for Individual Treatment Effects

This repository contains the code accompanying the paper:

**“Advantages and Limitations in the Use of Transfer Learning for Individual Treatment Effects in Causal Machine Learning”**  
by *Aydin & Brandt*.


---

## Repository Structure


### **Data**
- **Simulation** (`Data_generation/Simulation.R`):  
  Generates simulated datasets.

- **Empirical Example** (`Data_generation/IHDS.R`):  
  Prepares and subsets the [IHDS-II household survey dataset](https://ihds.umd.edu/data/ihds-2) for empirical analysis.

### **Functions (Model & Training Procedure)**
- **TARNet Model** (`TARNet.py`):  
  Defines the shared representation and two potential outcome heads.

- **Phase 1 – Distribution Alignment** (`Optimize_IPM.py`):  
  Trains the representation to align source and target treatment/control distributions using an Integral Probability Metric (IPM).

- **Phase 2 – Factual Loss Training** (`Optimize_Loss.py`):  
  Trains the treatment and control outcome heads on the target dataset.

### **Distance Measures**
- **Distribution Distances** (`Distances/Distance.py`):  
  Implements Wasserstein/IPM-based metrics for quantifying dataset distribution differences.

---

## Contact

For questions or discussion, feel free to open an issue or contact the authors.

