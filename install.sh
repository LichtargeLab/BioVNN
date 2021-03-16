#!/usr/bin/env bash

# check for conda and install if needed. Will download python 3.6. Also, could switch this to miniconda
if ! which conda > /dev/null; then
   echo -e "Conda not found! Install? (y/n) \c"
   read REPLY
   if [ "$REPLY" = "y" ]; then
      wget https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh
      bash ~/anaconda.sh -b -p $HOME/anaconda3
      echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc # add anaconda bin to the environment
      export PATH="$HOME/anaconda3/bin:$PATH"
   fi
fi

function check_download_file()
{
    FILENAME=$1
    MD5=$2
    CHECK_MD5SUM=$3
    if [ -f $FILENAME ]
    then
        if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
        then
            echo $(date) "$FILENAME was downloaded."
        else
            echo $(date) "Downloading $FILENAME"
            curl -O http://static.lichtargelab.org/BioVNN/$FILENAME
        fi
    else
        echo $(date) "Downloading $FILENAME"
        curl -O http://static.lichtargelab.org/BioVNN/$FILENAME
    fi

    if [[ `$CHECK_MD5SUM $FILENAME` = *"$MD5"* ]]
    then
        echo $(date) "Decompressing $FILENAME"
        tar zxvf $FILENAME -C ./
    else
        return
    fi
}

CHECK_MD5SUM=`command -v md5sum`
if [[ $CHECK_MD5SUM != *"md5sum"* ]]
then
    CHECK_MD5SUM=`command -v md5`
    if [[ $CHECK_MD5SUM != *"md5"* ]]
    then
        echo $(date) "md5 or md5sum is required to check file."
        exit 1
    fi
fi

# download data
check_download_file BioVNN_data.tar.gz a5b08d9027fcd2f7c0ec8d280459bb9f $CHECK_MD5SUM

# install env
ENV_NAME='BioVNN'

ENVS=$(conda env list | awk '{print $1}' )
if [[ $ENVS = *$ENV_NAME* ]]; then
   conda activate $ENV_NAME
else
    # make virtual environment
    conda env create -f environment.yml --name $ENV_NAME
    conda activate $ENV_NAME
fi;

# Set-up .env
if [ ! -f ./.env ]; then
    # Make the file
    touch ./.env

    # Record where this repo is:
    CURRENT=$(pwd)
    echo "Type the location of this repository (default:$CURRENT):"
    read input
    dotenv -f .env set REPO_DIR ${input:-$CURRENT}
    if [[ $input ]]; then
        REPO_DIR=$input
    else
        REPO_DIR=$CURRENT
    fi
    # Record the results folder destination
    echo "Type the location you want to store results (default:${REPO_DIR}/results/):"
    read input
    dotenv -f .env set RESULTS_DIR ${input:-${REPO_DIR}/results/}

    # Record the data folder destination
    echo "Type the data location (default:${REPO_DIR}/data/):"
    read input
    dotenv -f .env set DATA_DIR ${input:-${REPO_DIR}/data/}
    DATA_DIR=${input:-${REPO_DIR}/data/}
    # record where the depmap data is stored
    DEPMAP_VER=19Q3
    dotenv -f .env set DEPMAP_VER ${DEPMAP_VER}
    dotenv -f .env set DEPMAP_DIR ${DATA_DIR}/DepMap/${DEPMAP_VER}/

fi

echo $(date) "Installation was completed."
