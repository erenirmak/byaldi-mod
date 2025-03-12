"""
Issue:
The original implementation of the `byaldi` library had a dependency on `poppler-utils` for PDF processing functionality. This dependency required external system-level installations, which could lead to compatibility issues and additional setup complexity.

Solution:
To address this issue, the dependency on `poppler-utils` was replaced with `PyMuPDF`, a Python library that provides similar PDF processing capabilities. The classes in `byaldi` that relied on `poppler-utils` were overridden to use `PyMuPDF` instead. This change simplifies the setup process by removing the need for external system-level dependencies and leverages a pure Python solution for PDF processing.

Steps Taken:
1. Identified the classes and methods in `byaldi` that used `poppler-utils`.
2. Replaced the `poppler-utils` functionality with equivalent methods from `PyMuPDF`.
3. Tested the updated implementation to ensure that all PDF processing features work as expected with `PyMuPDF`.

Benefits:
- Simplified installation and setup process.
- Reduced dependency on external system-level tools.
- Improved compatibility across different environments.
"""