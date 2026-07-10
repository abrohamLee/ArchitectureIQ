ArchitectureIQ Quiz - Offline Static Version

How to use
1. Unzip ArchitectureIQ-quiz.zip.
2. Open the ArchitectureIQ-quiz folder.
3. Double-click index.html.
4. Use Next/Random or the question dropdown to navigate.
5. Select an answer to reveal the correct answer, ranked metrics, and learning curves.

Notes
- Works offline after unzipping.
- No Python, Streamlit, PyTorch, or repository checkout is required for quiz users.
- Supported target users: Windows and macOS users with a modern browser.
- Answers are embedded in the local static files, so this package is for practice, demos, or teaching, not for hidden-answer exams.

Maintainer rebuild command
python tools/question_static_exporter/export.py --data-root data --out outputs/ArchitectureIQ-quiz --zip outputs/ArchitectureIQ-quiz.zip --overwrite
