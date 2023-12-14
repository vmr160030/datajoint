# datajoint
Datajoint pipieline for Rieke Lab

## Docker
1. Install datajoint package: `conda install -c conda-forge datajoint`
2. Install and Run Docker Desktop: https://docs.docker.com/desktop/install/mac-install/
3. Set up datajoint mysql docker image: https://github.com/datajoint/mysql-docker
```mkdir mysql-docker
cd mysql-docker
wget https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml
docker-compose up -d
```

4. Run `datajoint_tutorial_Dec.ipynb` :)