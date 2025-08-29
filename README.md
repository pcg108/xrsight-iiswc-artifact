**1. Download this artifact repo**

<pre>
git clone https://github.com/pcg108/xrsight-iiswc-artifact.git
cd xrsight-iiswc-artifact
git submodule update --init --recursive
</pre>

**2. Install conda** 

<pre>
cd ~
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"

# Select yes when prompted to update profile
bash Miniforge3-$(uname)-$(uname -m).sh

source ~/.bashrc
</pre>

**3. Chipyard setup instructions:**

<pre>
cd xrsight-chipyard
./build-setup.sh riscv-tools -s 5
</pre>

**4. Firesim setup:**

Edit your ~/.bashrc file so that the following section is removed:
<pre>
# If not running interactively, don't do anything
case $- in
     *i*) ;;
       *) return;;
esac
</pre>

<pre>
cd ~/.ssh
ssh-keygen -t ed25519 -C "firesim.pem" -f firesim.pem
[create passphrase] (blank)
</pre>

<pre>
cd ~/.ssh
cat firesim.pem.pub >> authorized_keys
chmod 0600 authorized_keys
</pre>

<pre>
cd ~/.ssh && ssh-agent -s > AGENT_VARS && source AGENT_VARS && ssh-add firesim.pem && cd -
cd xrsight-chipyard/sims/firesim
source sourceme-manager.sh
</pre>

<pre>
# Add to ~/.bashrc (necessary for firesim), but DO NOT SOURCE
source /ecad/tools/xilinx/Vivado/2022.1/settings64.sh
source /ecad/tools/vlsi.bashrc
</pre>

<pre>
cd xrsight-chipyard/software/firemarshal
./init-submodules
./marshal -v build ubuntu-base.json
./marshal -v install ubuntu-base.json
</pre>

**5. Install ILLIXR host worker**

<pre>
conda install -c conda-forge glew boost-cpp spdlog eigen libvulkan-headers libvulkan-loader vulkan-tools yaml-cpp glfw glm libgl
cd xrsight-iiswc-artifact/ILLIXR && mkdir build && cd build

export OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
cmake .. -DCMAKE_INSTALL_PREFIX=/home/eval-iiswc-1/illixr-deps/  -DYAML_FILE=profiles/gpu_offload.yaml -DCMAKE_BUILD_TYPE=Release 
</pre>

<pre>
# Add to ~/.bashrc:
export PATH=$PATH:/home/eval-iiswc-1/illixr-deps/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/eval-iiswc-1/illixr-deps/lib
</pre>

<pre>
cmake --build . -j32 && cmake --install .
</pre>

**6. Extract linux images from tar.gz**

<pre>
./download_images.sh
</pre>

**7. Run experiments:**

VIO:
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-exp1.sh
./setup-exp1.sh
</pre>

In a new terminal, login and run 
<pre>
chmod +x monitor-exp1.sh
./monitor-exp1.sh
</pre>

Eye Tracking:
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-exp2.sh
./setup-exp2.sh
</pre>
In a new terminal, login and run 
<pre>
chmod +x monitor-exp2.sh
./monitor-exp2.sh
</pre>

GTSAM:
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-exp3.sh
./setup-exp3.sh
</pre>

In a new terminal, login and run
<pre>
chmod +x monitor-exp3.sh
./monitor-exp3.sh
</pre>

8. Generate plots
<pre>
cd ~/xrsight-iiswc-artifact
python parse.py
</pre>

**9. VIO experiments**

These scripts help generate Figures 12-14 in the paper, for the VIO case study.

Baseline (~4-5 hours):
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-vio-generic.sh
./setup-vio-generic.sh
</pre>

In a new terminal, login and run
<pre>
chmod +x monitor-vio-generic.sh
./monitor-vio-generic.sh
</pre>

Saturn (~4-5 hours):
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-vio-saturn.sh
./setup-vio-saturn.sh
</pre>

In a new terminal, login and run
<pre>
chmod +x monitor-vio-saturn.sh
./monitor-vio-saturn.sh
</pre>

Gemmini (note that this will likely take 3-4x as long):
<pre>
cd ~/xrsight-iiswc-artifact
chmod +x setup-vio-gemmini.sh
./setup-vio-gemmini.sh
</pre>

In a new terminal, login and run
<pre>
chmod +x monitor-vio-gemmini.sh
./monitor-vio-gemmini.sh
</pre>

**10. Generate plots**
<pre>
# for all 3 configurations
python parse_vio_full.py

# for baseline and saturn only
python parse_vio_saturn_only.py
</pre>

