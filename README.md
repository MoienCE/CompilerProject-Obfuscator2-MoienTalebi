# CompilerProject-Obfuscator-MoienTalebi
A college project for a compiler design course.

HEAD Moien Talebi,

Arian khalili 40117913,

parsa parhami  40116583,

moien talebi  40124563






#  Mini-C Obfuscator

A Python-based tool that performs source-level obfuscation on a simplified version of the C language (Mini-C). Built for educational and research purposes.

---

##  Features

-  **CLI Interface** with flexible obfuscation flags
-  **Obfuscation Techniques**:
  - Rename Variables
  - Remove Dead Code
  - Insert Opaque Predicates
-  **Performance Metrics**: Original vs. Obfuscated size & time
-  **AST Visualization** with Graphviz
-  **Automated Testing** via `pytest`
-  **GitHub Actions CI** for auto-testing and quality control

---

##  Getting Started

###  Requirements

- Python 3.10+
- Graphviz (for `dot` command):
  - Windows: Install from [https://graphviz.org/download/](https://graphviz.org/download/)
  - Add Graphviz `bin` to your system PATH.

###  Install dependencies

```bash
pip install -r requirements.txt
>>>>>>> develop
