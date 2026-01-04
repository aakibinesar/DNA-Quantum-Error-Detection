# DNA-Based Quantum Error Detection via Biochemical Structure
Complete Pauli error detection in DNA-labeled quantum channels using biochemical structure (100% detection, zero quantum overhead) - ISIT 2026


This repository contains the complete simulation code for the paper:

**"X-Error Detection and Pauli Characterization via Biochemical Structure in DNA-Labeled Quantum Channels"**  
*Aakib Bin Nesar, North South University*  
Submitted to IEEE ISIT 2026

## Overview

We exploit DNA's biochemical properties (Watson-Crick complementarity, purine/pyrimidine classification, hydrogen bonding) for complete Pauli error detection in quantum channels, achieving **100% error characterization with zero quantum overhead**.

### Key Results

- **Complete Pauli detection**: X, Y, and Z errors have unique biochemical signatures
- **100% detection rate**: Multi-property framework achieves perfect error identification
- **Capacity gain**: 3×H₂(p/3) for combined detection (48-1227 millibits for p∈[0.01,0.30])
- **Zero quantum overhead**: Uses classical biochemical measurements only

## Requirements
```bash
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.8+
- NumPy
- Matplotlib

## Usage

Run all experiments:
```bash
python DNA_Quantum_Error_Detection_Channel.py
```

This generates:
- `fig1_capacity.pdf` - Capacity comparison and gain
- `fig2_complete_detection.pdf` - Complete Pauli characterization (X, Y, Z detection)
- `fig3_detection_rate.pdf` - Single-property detection validation

## Quick Start Example
```python
import numpy as np
from Complementarity_Channel import multi_property_detection

# Test complete Pauli detection at p=0.15
labeling_id = 4
p_x = p_y = p_z = 0.05  # symmetric noise (total p=0.15)

results = multi_property_detection(labeling_id, p_x, p_y, p_z, n_trials=10000)

print(f"X-error detection: {results['x_detection_rate']:.1%}")  # → 100%
print(f"Y-error detection: {results['y_detection_rate']:.1%}")  # → 100%
print(f"Z-error detection: {results['z_detection_rate']:.1%}")  # → 100%
```

## Experiments

The code includes 5 experiments validating theoretical predictions:

1. **Pauli Classification**: Verifies X-errors map to Watson-Crick complements
2. **Capacity Analysis**: Compares baseline vs complementarity-aware capacity
3. **Complete Detection Framework**: Validates 100% detection for X, Y, Z errors
4. **Protocol Simulation**: End-to-end detection rate validation (5000 trials)
5. **Flag Matrix Analysis**: Confirms biochemical signature uniqueness

## Key Functions

- `multi_property_detection()`: Simulates detection using all three biochemical properties
- `detect_x_error_watson_crick()`: X-error detection via complementarity
- `detect_y_error_purine_pyrimidine()`: Y-error detection via base classification  
- `detect_z_error_hydrogen_bonds()`: Z-error detection via bonding strength
- `capacity_with_complementarity()`: Computes capacity gain from side information

## Results Validation

All simulations match theoretical predictions:
- X-error detection: 100.0% (Theory: 100%)
- Y-error detection: 100.0% (Theory: 100%)  
- Z-error detection: 100.0% (Theory: 100%)
- Single-property detection: 33.3% (Theory: 1/3)

## Citation

If you use this code, please cite:
```bibtex
@inproceedings{nesar2026dna,
  title={X-Error Detection and Pauli Characterization via Biochemical Structure in DNA-Labeled Quantum Channels},
  author={Nesar, Aakib Bin},
  booktitle={IEEE International Symposium on Information Theory (ISIT)},
  year={2026}
}
```

## License

MIT License - see LICENSE file

## Contact

Aakib Bin Nesar  
Department of Electrical and Computer Engineering  
North South University, Dhaka, Bangladesh  
Email: aakib.nesar@northsouth.edu

## Acknowledgments

This work was conducted at North South University as part of a master's thesis on DNA-based quantum information processing.
```

---

## **requirements.txt**
```
numpy>=1.21.0
matplotlib>=3.4.0
```

---

## **LICENSE (MIT License)**
```
MIT License

Copyright (c) 2026 Aakib Bin Nesar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
