install_conda_and_create_env() {
    local miniconda_dir="$HOME/miniconda3"
    
    if [ ! -d "$miniconda_dir" ]; then
        echo "Installing Miniconda..."
        # Install Miniconda
        mkdir -p "$miniconda_dir"
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$miniconda_dir/miniconda.sh"
        bash "$miniconda_dir/miniconda.sh" -b -u -p "$miniconda_dir"
        rm -f "$miniconda_dir/miniconda.sh"

        echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
        export PATH="$HOME/miniconda3/bin:$PATH"
        
        eval "$($miniconda_dir/bin/conda shell.bash hook)"
    else
        echo "Miniconda is already installed."
        export PATH="$HOME/miniconda3/bin:$PATH"
    fi

    if [ -f "$miniconda_dir/etc/profile.d/conda.sh" ]; then
        . "$miniconda_dir/etc/profile.d/conda.sh"
    else
        echo "Error: Conda initialization script not found!"
        exit 1
    fi

    if conda env list | grep -q "^3101_proj"; then
        echo "Conda environment '3101_proj' already exists. Activating it..."
        conda activate 3101_proj
    else
        echo "Creating Conda environment '3101_proj'"
        conda create --name 3101_proj python=3.9 -y
        conda activate 3101_proj
        conda install ipykernel -y
    fi
}

# Call the function
install_conda_and_create_env
ipython kernel install --user --name=3101_proj --display-name "(3101_proj)"
