# VMEC AI Assistant (uwplasma)

This project is an AI-powered assistant and interface for interacting with the `vmec_jax` codebase. It allows users to ask questions about the repository, run VMEC simulations, and visualize results through a simple web interface.

---

## Quick Start

After installing dependencies and starting Ollama, run:

```bash
python app.py
```

Then open your browser at:

```
http://127.0.0.1:5001
```

---

## Features

* AI assistant using Ollama (LLaMA3)
* Run VMEC simulations with custom parameters
* Generate plots (iota, phi)
* Web UI using Flask
* Context-aware responses based on repository files

---

## Important Setup Notes

### Dependencies

This project uses external libraries that are not included in the repository.

Install them with:

```bash
pip install vmec_jax flask flask-cors matplotlib ollama
```

---

### Ollama Setup (Required)

This project uses an Ollama model (LLaMA3).

1. Install Ollama: https://ollama.com
2. Start the model:

```bash
ollama run llama3
```

Make sure this is running before starting the application.

---

### Generated Files

Some files are created automatically when you run the project.
You do not need to create them manually.

These include:

* `input_generated`
* `*.png` (plots like iota, phi)
* `wout_generated.nc`

They will appear after running a simulation.

---

### Optional: Virtual Environment

Run these commands inside the project folder:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## How to Use

### Web Interface

1. Start the app:

   ```bash
   python app.py
   ```
2. Open the browser at:

   ```
   http://127.0.0.1:5001
   ```
3. Ask questions or run simulations using the UI.

---

### Terminal Mode

Run the assistant directly:

```bash
python assistant.py
```

Then type queries like:

```
How do I run vmec_jax?
```

---

## Simulation

You can run a VMEC simulation by entering parameters in the UI:

* `ns`
* `mpol`
* `ntor`

The system will:

1. Generate a modified input file
2. Run the VMEC simulation
3. Produce output (`wout_generated.nc`)
4. Generate plots (e.g., iota and phi profiles)

---

## Project Structure

```
.
├── app.py
├── assistant.py
├── index.html
├── README.md
│
├── input.circular_tokamak
├── wout_summary.txt
├── vmec_readme.md
│
├── showcase_axisym_input_to_wout.py
└── .gitignore
```

---

## Notes

* Plots are saved as PNG files
* Errors (if any) will appear in the terminal
* Make sure Ollama is running before using the assistant

---

## Author

Developed as part of UW Plasma research work.

