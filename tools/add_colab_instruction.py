from pathlib import Path
import glob, json

build_script = "!curl -O https://raw.githubusercontent.com/aws-samples/d2l-java/master/tools/colab_build.sh && bash colab_build.sh"
fix_gpu_script = "!curl -O https://raw.githubusercontent.com/aws-samples/d2l-java/01c3fcdcae62a13f363a04b87ef6ff08e64a2ebe/tools/fix-colab-gpu.sh && bash fix-colab-gpu.sh"
script_constructor = {"cell_type": "code", "metadata": {}, "outputs": [], "source": [build_script], "execution_count": None}
fix_gpu_constructor = {"cell_type": "code", "metadata": {}, "outputs": [], "source": [fix_gpu_script], "execution_count": None}

md = [
    "## Prepare Java Kernel for Google Colab\n",
    "Since Java is not natively supported by Colab, we need to run the following code to enable Java kernel on Colab.\n",
    "\n",
    "1. Run the cell bellow (click it and press Shift+Enter),\n",
    "2. (If training on CPU, skip this step) If you want to use the GPU with MXNet in DJL 0.10.0, we need CUDA 10.1 or CUDA 10.2.\n"
    "Since Colab supports CUDA 10.1, we will have to follow some steps to setup the environment.\n"
    "Refresh the page (press F5) and stay at Python runtime on GPU. Run the file fix-colab-gpu script.\n"
    "\n"
    "And then ensure that you have switched to CUDA 10.1.\n"
    "3. After that, switch runtime to Java and hardware to GPU.(Might require refreshing the page and switching runtime)\n",
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
        data["cells"].insert(0, fix_gpu_constructor)
        writer.write(json.dumps(data))
