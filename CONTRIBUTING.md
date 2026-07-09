# Contributing to PapsAI XNet

First and foremost, thank you for your interest in contributing to **PapsAI XNet**.

PapsAI XNet is an open-source explainable artificial intelligence (XAI) framework for automated cervical cytology classification. It forms part of the broader **PapsAI Digital Microscopy Ecosystem**, an ongoing initiative dedicated to developing affordable, trustworthy, and clinically deployable AI solutions for cervical cancer screening, particularly in low-resource settings.

We welcome contributions from researchers, clinicians, biomedical engineers, computer scientists, students, and the wider open-source community.

---

# Our Mission

Our goal is to build transparent, reproducible, and clinically meaningful AI technologies that improve cervical cancer diagnosis while promoting open scientific collaboration.

We encourage contributions that improve:

- Model performance
- Explainability
- Computational efficiency
- Documentation
- Clinical usability
- Software quality
- Reproducibility

---

# Ways to Contribute

You can contribute in many ways, including:

## Bug Reports

Please report:

- software bugs
- installation problems
- incorrect outputs
- documentation errors

Use GitHub Issues and include:

- operating system
- Python version
- package versions
- steps to reproduce
- expected behaviour
- screenshots (if applicable)

---

## Feature Requests

We welcome suggestions for:

- new explainability methods
- new CNN architectures
- improved ANFIS modules
- optimization methods
- deployment features
- visualization tools
- user interface improvements

Please describe:

- the problem
- the proposed solution
- expected benefits

---

## Code Contributions

Typical contributions include:

- fixing bugs
- improving algorithms
- adding tests
- improving documentation
- optimizing performance
- refactoring code

---

## Research Contributions

Researchers are encouraged to contribute:

- benchmark studies
- additional datasets
- external validation
- explainability analyses
- ablation studies
- computational optimization
- clinical evaluation

---

# Development Workflow

1. Fork the repository.

2. Clone your fork.

```bash
git clone https://github.com/<your-username>/PapsAI-XNet.git
```

3. Create a new branch.

```bash
git checkout -b feature/my-feature
```

4. Install the development environment.

```bash
conda env create -f environment.yml

conda activate papsai-xnet
```

5. Implement your changes.

6. Test your implementation.

7. Commit your changes.

```bash
git commit -m "Add feature description"
```

8. Push your branch.

```bash
git push origin feature/my-feature
```

9. Submit a Pull Request.

---

# Coding Standards

Please follow:

- PEP 8
- meaningful variable names
- clear documentation
- modular code
- reproducible experiments

Use descriptive commit messages.

---

# Documentation

New functionality should include:

- inline comments
- docstrings
- README updates (where applicable)
- usage examples

---

# Testing

Before submitting a Pull Request:

- verify installation
- ensure training executes successfully
- ensure inference executes successfully
- verify evaluation metrics
- confirm no existing functionality is broken

---

# Pull Requests

Please include:

- description of the contribution
- motivation
- related issue number (if applicable)
- testing performed
- screenshots (if UI related)

Pull requests should remain focused on a single feature or bug fix.

---

# Data

Please do **not** upload:

- copyrighted datasets
- patient data
- personally identifiable information
- clinical records

Only use datasets that can legally be redistributed or referenced.

---

# Code of Conduct

We are committed to maintaining an open, inclusive, respectful, and collaborative research environment.

Please:

- be respectful
- provide constructive feedback
- welcome new contributors
- acknowledge previous work
- maintain professional communication

Harassment, discrimination, or abusive behaviour will not be tolerated.

---

# Licensing

By contributing to this repository, you agree that your contributions will be released under the **Apache License 2.0**.

---

# Citation

If your contribution supports published work, please cite the corresponding publication.

```
Wasswa W., et al.

Explainable Hybrid Deep Learning for Automated Cervical Cytology Classification.

Nature Scientific Reports.
```

---

# Roadmap

Current research directions include:

- Whole-slide image analysis
- PapsAI XNet V2
- Self-supervised learning
- Vision Transformers
- Multimodal AI
- Active learning
- Clinical deployment
- Edge AI optimization
- Federated learning
- Digital pathology integration

---

# Contact

Project Lead

**Dr. William Wasswa**

Faculty of Applied Sciences and Technology

Mbarara University of Science and Technology

Uganda

Website:

https://papsai.org

Email:wwasswa@must.ac.ug


---

# Acknowledgements

We sincerely appreciate every contribution made to PapsAI XNet.

Together we can build transparent, trustworthy, and accessible AI technologies that improve cervical cancer screening worldwide.
