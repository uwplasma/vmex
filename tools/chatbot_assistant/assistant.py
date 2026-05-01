import ollama
import vmec_jax as vj
print(dir(vj))
import json
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

with open ("vmec_readme.md") as f:
    readme = f.read()
with open ("showcase_axisym_input_to_wout.py") as f:
    example = f.read()
with open ("input.circular_tokamak") as f:
    input_file = f.read()

CONTEXT_MAP = {
    "README": ("README.md", readme),
    "SCRIPT": ("showcase_axisym_input_to_wout.py", example),
    "INPUT": ("input.circular_tokamak", input_file),
}

def decide(question):
    response = ollama.chat(
        model = 'llama3',
       messages=[
            {
                "role": "system",
                "content": """
You are an AI agent controller.

Decide:
1. Action:
   - "answer"
   - "run_vmec"

2. Parameters (if action is run_vmec):
   - ns (integer or null)
   - mpol (integer or null)
   - ntor (integer or null)

3. Context:
   - README
   - SCRIPT
   - INPUT
   - OUTPUT

Respond ONLY in JSON:
{
  "action": "...",
  "ns": number or null,
  "mpol": number or null,
  "ntor": number or null,
  "context": "..."
}
"""
            },
            {'role':"user", "content":question}
         ]
    )
    try: 
        return json.loads(response['message']['content'])
    except:
        return {"action": "answer", "context" : "README"}
def update_input_file(ns, mpol, ntor):
    with open("input.circular_tokamak") as f:
        text = f.read()
    if ns:
        text = text.replace("ns = 64", f"ns = {ns}")
    if mpol:
        text = text.replace("mpol = 5", f"mpol = {mpol}")
    if ntor:
        text = text.replace("ntor = 3", f"ntor = {ntor}")
    with open("input_generated", "w") as l:
        l.write(text)
    return "input_generated"

def run_vmec(ns=None, mpol=None, ntor=None):
    ns = int(ns) if ns else 64
    mpol = int(mpol) if mpol else 5
    ntor = int(ntor) if ntor else 3

    try:
        print("RUN_VMEC STARTED")

        input_path = update_input_file(ns, mpol, ntor)

        run = vj.run_fixed_boundary(input_path, max_iter=20)

        out_path = Path('wout_generated.nc')
        vj.write_wout_from_fixed_boundary_run(out_path, run, include_fsq=True)

        wout = vj.load_wout(out_path)

        print("ABOUT TO PLOT")

        # 🔥 TEST PLOT FIRST (guaranteed to work)
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.savefig("test.png")
        plt.close()

        # 🔥 REAL PLOTS
        plt.figure()
        plt.plot(wout.phi)
        plt.title("Phi profile")
        plt.savefig("phi.png")
        plt.close()

        plt.figure()
        plt.plot(wout.iotaf)
        plt.title("Iota profile")
        plt.savefig("iota.png")
        plt.close()

        print("FILES AFTER SAVE:", os.listdir())

        plots = [f for f in os.listdir() if f.endswith(".png")]

        return {
            "text": f"aspect: {wout.aspect}",
            "plots": plots
        }

    except Exception as e:
        print("ERROR:", e)
        raise e   # 🔥 DO NOT hide errors
        
system_prompt = """
You are a scientific assistant for the vmec_jax repository.

RULES:
- Answer ONLY using the provided context.
- ALWAYS mention the source file in the first sentence (e.g., "From README.md").
- Do NOT use external knowledge.
- Do NOT invent values, functions, or explanations.

BEHAVIOR:
- If the answer is not clearly present in the context, say exactly:
  "I don't see this in the provided files"
- If values are present, restate them clearly without adding interpretation.
- For workflow questions, refer only to steps explicitly shown in the script.
- For input questions, use only input.circular_tokamak.
- For output questions, use only the provided output summary.

STYLE:
- Be concise and precise.
- Do not add unnecessary explanations.
"""
def handle_query(question):
    decision = decide(question)
    action = decision.get("action","answer")
    if action == "run_vmec":
        ns = decision.get("ns")
        mpol = decision.get("mpol")
        ntor = decision.get("ntor")
        return run_vmec(ns, mpol, ntor)
    context_key = decision.get("context","README") 
    filename, context = CONTEXT_MAP.get(context_key, CONTEXT_MAP['README'])
    response = ollama.chat(
        model = 'llama3',
        messages = [
            {'role':'system','content':system_prompt},
            {'role':'user','content': f'{context}\n\nQuestion: {question}'}
        ]
    )
    return response['message']['content']
if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() == "exit":
            break

        decision = decide(q)
        print("Decision:", decision)  

        answer = handle_query(q)
        print("\nAnswer:\n", answer)
        print("\n" + "-"*40 + "\n")