# Datajoint Pipeline
Datajoint pipieline for Rieke Lab

## Dependencies
Install the following in your conda environment:
- visionloader (from `artificial_retina_software_pipeline`)
- symphony_data (from Mike's `MEA/src/analysis`)
    - Ensure that you have a `~/Documents/custom_config.py` file defining your NAS array paths appropriately.

Pull Samarjit's datajoint repo:
- https://github.com/SamarjitK/datajoint
- I'm generally using my `dev_vr` branch in this. Generally the major updates should have been merged to main.

Update default paths in Samarjit's repo if needed:
- Ensure that in `next-app/api/helpers/utils.py`, `NAS_DATA_DIR` and `NAS_ANALYSIS_DIR` are set appropriately for you.

## Pandas and numpy versioning
Newer (>=2.2) versions of the pandas package (which is used heavily for data tables) have issues interfacing with numpy, giving an error that may look like this:
```
File lib.pyx:2538, in pandas._libs.lib.maybe_convert_objects()

TypeError: Cannot convert numpy.ndarray to numpy.ndarray
```

To fix this, downgrade pandas to an earlier version. For me, pandas 2.0.3 and numpy 1.24.3 work fine. To downgrade pandas, run the following in terminal with your conda env activated:
`conda install -c conda-forge pandas=2.0.3`


## Datajoint and Docker setup
1. Install latest datajoint package (v0.14.3 as of March 2025):
    1. Activate your conda env in terminal
    2. Follow installation isntructions from their readme. For v0.14.3, they are: `conda install -c conda-forge datajoint`
2. Install and Run Docker Desktop: https://docs.docker.com/desktop/install/mac-install/
3. Set up datajoint mysql docker image: https://github.com/datajoint/mysql-docker
```mkdir mysql-docker
cd mysql-docker
wget https://raw.githubusercontent.com/datajoint/mysql-docker/master/docker-compose.yaml
docker compose up -d
```

4. Run latest tutorial .ipynb under `nbs/` :)