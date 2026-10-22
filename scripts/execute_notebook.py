import json
import io
import base64
import contextlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_notebook():
    nb_path = Path("parking_violation_detection.ipynb")
    assert nb_path.exists(), "Notebook not found"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    global_ns = {}
    exec_count = 1

    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell["source"])
        print(f"Executing code cell {exec_count} (index {idx})...")
        stdout_buf = io.StringIO()
        cell["outputs"] = []
        captured_figs = []

        def custom_show(*args, **kwargs):
            for fignum in plt.get_fignums():
                fig = plt.figure(fignum)
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight")
                img_buf.seek(0)
                img_b64 = base64.b64encode(img_buf.read()).decode("utf-8")
                captured_figs.append(img_b64)
            plt.close("all")

        orig_show = plt.show
        plt.show = custom_show

        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(source, global_ns)
            custom_show()
        except Exception as e:
            print(f"Error executing cell {exec_count}: {e}")
            cell["outputs"].append({
                "output_type": "error",
                "ename": type(e).__name__,
                "evalue": str(e),
                "traceback": [str(e)]
            })
        finally:
            plt.show = orig_show

        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            cell["outputs"].append({
                "output_type": "stream",
                "name": "stdout",
                "text": stdout_text.splitlines(keepends=True)
            })

        for b64 in captured_figs:
            cell["outputs"].append({
                "output_type": "display_data",
                "data": {
                    "image/png": b64,
                    "text/plain": "<Figure size 900x500 with 1 Axes>"
                },
                "metadata": {}
            })

        cell["execution_count"] = exec_count
        exec_count += 1

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    print("Notebook executed and saved successfully with all outputs!")

if __name__ == "__main__":
    run_notebook()
