from pathlib import Path
import glob, json

build_script = "!curl -O https://raw.githubusercontent.com/deepjavalibrary/d2l-java/master/tools/colab_build.sh && bash colab_build.sh"
script_constructor = {"cell_type": "code", "metadata": {}, "outputs": [], "source": [build_script], "execution_count": None}

md = [
    "## Prepare Java Kernel for Google Colab\n",
    "Since Java is not natively supported by Colab, we need to run the following code to enable Java kernel on Colab.\n",
    "\n",
    "1. Run the cell bellow (click it and press Shift+Enter),\n",
    "2. then refresh the page (press F5) right after that.\n",
    "\n",
    "Now you can write Java code."
   ]
instruction_constructor = {"cell_type": "markdown", "metadata": {}, "source" : md}

for file in Path('.').glob('**/*.ipynb'):
    with open(file, mode= "r", encoding= "utf-8") as f:
        data = json.loads(f.read())
    with open(file, 'w') as writer:
        data["cells"].insert(0, script_constructor)
        data["cells"].insert(0, instruction_constructor)
        writer.write(json.dumps(data))
