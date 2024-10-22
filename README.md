# Datajoint Pipeline
Datajoint pipieline for Rieke Lab

## Dependencies
Install the following in your conda environment:
- visionloader (from `artificial_retina_software_pipeline`)
- symphony_data (from Mike's `MEA/src/analysis`)

## Datajoint and Docker setup
1. Install latest datajoint package (v0.14.1 as of Feb 2024):
    1a. Pull this gitrepo to a local dir: https://github.com/datajoint/datajoint-python and cd to it
    1. Activate your conda env in terminal
    2. Follow installation isntructions from their readme. For v0.14.1, they are:
        1. `pip install -r requirements.txt`
        2. `python setup.py install`
2. Install and Run Docker Desktop: https://docs.docker.com/desktop/install/mac-install/
3. Set up datajoint mysql docker image: https://github.com/datajoint/mysql-docker
```mkdir mysql-docker
cd mysql-docker
wget https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml
docker-compose up -d
```

4. Run latest tutorial .ipynb under `nbs/` :)