from pathlib import Path
import glob, json, os

constructor = {"cell_type": "markdown", "metadata": {}}
binder_prefix = "Run this notebook online:[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deepjavalibrary/d2l-java/master?filepath="
colab_prefix = "Colab: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepjavalibrary/d2l-java/blob/colab/"

for file in Path('.').glob('**/*.ipynb'):
    if os.path.basename(file) not in ["index.ipynb", "djl-imports.ipynb", "plot-utils.ipynb"]:
        with open(file, mode= "r", encoding= "utf-8") as f:
            data = json.loads(f.read())
        with open(file, 'w') as writer:
            constructor["source"] = [binder_prefix + str(file) + ") or " + colab_prefix + str(file) + ")"]
            if data["cells"][0] != constructor:
                data["cells"].insert(0, constructor)
            writer.write(json.dumps(data))
