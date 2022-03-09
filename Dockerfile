# Approximately 10 min to build

FROM nvidia/cuda:10.2-cudnn7-devel
# Python
ARG python_version=3.7
ARG SSH_PASSWORD=password
ARG SSH_PUBLIC_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCojldgho9VK4WaCbSjBAPr9i6daYdQ5s9uGpVuXLI6cAKtzT8G9AQg+wYZayYNthexuzmp5BwpyJT8QQTsUUgBuaocSAjZff8uFKNN9yVMVtT8RIYw/NVVkb97ZPx3ZxN2e7m6BlJyKNg8jKOw4qiUMCH70wYprjEKVUzEjJnM7Mq/BnJPYJr+DQG7IE9uGJwGiE7gHAatsECkcg+QcrMHpLwtha91VE/U13C5dSE072mAX50QnWSGZV2SGg+o8AJViwixJCNMZhld6thClmFezYJjsb9Uz1Hss6xatntxIjUmjL2Lyc/uWFiep+0/R5GPQ9Tbq929IpZ1DwbW5J0x rinatmullahmetov@Rinats-MacBook-Pro.local"


# https://docs.docker.com/engine/examples/running_ssh_service/
# Last is SSH login fix. Otherwise user is kicked off after login
RUN apt-get update && apt-get install -y openssh-server && \
    mkdir /var/run/sshd && echo "root:$SSH_PASSWORD" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    echo "export VISIBLE=now" >> /etc/profile && \
    mkdir /root/.ssh && chmod 700 /root/.ssh && \
    echo "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys && \
    chmod 644 /root/.ssh/authorized_keys

ENV NOTVISIBLE "in users profile"
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

# writing env variables to /etc/profile as mentioned here:
# https://docs.docker.com/engine/examples/running_ssh_service/#environment-variables
RUN echo "export CONDA_DIR=$CONDA_DIR" >> /etc/profile && \
    echo "export PATH=$CONDA_DIR/bin:$PATH" >> /etc/profile && \
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" >> /etc/profile && \
    echo "export LIBRARY_PATH=/usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH" >> /etc/profile && \
    echo "export CUDA_HOME=/usr/local/cuda" >> /etc/profile

# Install Miniconda
RUN mkdir -p $CONDA_DIR && \
    apt-get update && \
    apt-get install -y wget git vim htop zip libhdf5-dev g++ graphviz libgtk2.0-dev \
    openmpi-bin nano cmake libopenblas-dev liblapack-dev libx11-dev && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    ln /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/local/cuda/lib64/libcudnn.so && \
    ln /usr/lib/x86_64-linux-gnu/libcudnn.so.7 /usr/local/cuda/lib64/libcudnn.so.7 && \
    ln /usr/include/cudnn.h /usr/local/cuda/include/cudnn.h && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Install Data Science essential
RUN conda config --set remote_read_timeout_secs 100000.0 && \
    conda update openssl ca-certificates certifi && \
    conda install -y python=${python_version} && \
    pip install --upgrade pip --timeout=1000 && \
    pip install --upgrade requests --timeout=1000 && \
    conda install Pillow scikit-learn pandas matplotlib mkl nose pyyaml six && \
    pip install opencv-contrib-python requests scipy tqdm --timeout=1000 && \
    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch && \
    pip install pydantic graphviz hiddenlayer torchsummary --timeout=1000 && \
    pip install albumentations --timeout=1000 && \
    conda install -c anaconda jupyter && \
    conda install -c conda-forge jupyterlab && \
    pip install git+https://github.com/ipython-contrib/jupyter_contrib_nbextensions --timeout=1000 && \
    jupyter contrib nbextension install && \
    conda clean -yt

# Install NVIDIA Apex
RUN git clone https://github.com/NVIDIA/apex && cd apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
    cd .. && rm -r apex

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /usr/local/cuda/lib64:/lib/x86_64-linux-gnu:$LIBRARY_PATH
ENV CUDA_HOME /usr/local/cuda

COPY requirements.txt /jumanji/requirements.txt
RUN pip install -r /jumanji/requirements.txt

COPY . /jumanji

EXPOSE 8888 6006 22
ENTRYPOINT ["/usr/sbin/sshd"]
CMD ["-D"]
