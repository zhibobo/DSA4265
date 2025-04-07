### Running the script requires gitbash based on unix syntax
### also if u on windows, manually install mini-conda urself

install_conda_and_create_env() {
    # Detect OS type
    os_type=$(uname)
    echo "Detected OS: $os_type"

    # Check if conda is already installed
    if command -v conda &> /dev/null; then
        echo "Conda is already installed."
        # Automatically detect the installation directory of conda.
        miniconda_dir="$(dirname "$(dirname "$(which conda)")")"
        echo "Using conda installation at: $miniconda_dir"
    else
        echo "Conda not found."
        # For Linux, install Miniconda automatically
        if [[ "$os_type" == "Linux" ]]; then
            miniconda_dir="$HOME/miniconda3"
            echo "Installing Miniconda for Linux..."
            mkdir -p "$miniconda_dir"
            installer_url="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            wget -q "$installer_url" -O "$miniconda_dir/miniconda.sh"
            bash "$miniconda_dir/miniconda.sh" -b -u -p "$miniconda_dir"
            rm -f "$miniconda_dir/miniconda.sh"
            echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
            export PATH="$HOME/miniconda3/bin:$PATH"
        else
            # For Windows (Git Bash) automatic installation is more complex.
            echo "Cant install from script"
            echo "Install Miniconda manually from https://docs.conda.io/en/latest/miniconda.html"
            exit 1
        fi
    fi

    # Source the Conda initialization script using alternatives for Linux/Windows
    if [ -f "$miniconda_dir/etc/profile.d/conda.sh" ]; then
        . "$miniconda_dir/etc/profile.d/conda.sh"
    elif [ -f "$miniconda_dir/Scripts/activate" ]; then
        . "$miniconda_dir/Scripts/activate"
    else
        echo "Error: Conda initialization script not found!"
        exit 1
    fi

    # If the "dsa2465" environment exists, update it; otherwise, create it.
    if conda env list | grep -q "^dsa2465"; then
        echo "Conda environment 'dsa2465' already exists. Updating it..."
        conda activate dsa2465
        conda env update -f environment.yaml 
    else
        echo "Creating Conda environment 'dsa2465'..."
        conda create --name dsa2465 python=3.9 -y
        conda activate dsa2465
        conda env create -f environment.yaml

    fi
}

# Run the function
install_conda_and_create_env

# Register the IPython kernel for Jupyter
ipython kernel install --user --name=dsa2465 --display-name "(dsa2465)"