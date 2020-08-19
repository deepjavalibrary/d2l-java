# Troubleshooting

### 1. X11 error when running object detection notebooks on EC2 instances.

When you run object detection notebooks on EC2 instances via SSH, sometimes you may face X11 error when drawing bounding boxes.

```bash
java.awt.HeadlessException:
No X11 DISPLAY variable was set, but this program performed an operation which requires it.
java.awt.HeadlessException:
```
Here is how you can solve it:
* Use -X to ssh to your EC2 instance.

```bash
ssh -X -i "your_pem_file.pem" user-name@ip.address -L local-port-number:localhost:8888
```
follow this [guide](http://d2l.ai/chapter_appendix-tools-for-deep-learning/aws.html) for how to run notebook on EC2 instances in general.

* Install Xvfb package on your EC2 instance:
```bash
sudo apt-get install -y xvfb
```

* Run the following command to on a [tmux](https://github.com/tmux/tmux/wiki) session:
```bash
Xvfb :1
```

* Set `DISPLAY` variable to the same value where Xvfb is running.
```bash
export DISPLAY=:1
```

* Now start jupyter notebook normally and run the notebook
```bash
jupyter notebook
```

[Reference](https://stackoverflow.com/questions/10165761/java-cant-connect-to-x11-window-server-using-localhost10-0-as-the-value-of-t) 
