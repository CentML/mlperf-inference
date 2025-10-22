## How to run the docker compose file ##

**Offline mode**
```UID=$(id -u) GID=$(id -g) docker compose -f docker-compose-offline-mode.yml up```

**Server Mode**
```UID=$(id -u) GID=$(id -g) docker compose -f docker-compose-server-mode.yml up```

#### Evaluate accuracy ####

Create and activate a python virtual environment and then run these commands:

```pip install numpy pandas pyyaml transformers```

```python generate_total_val_output.py --mode [offline|server]```

```python main_eval_only.py --output_path total_val_output.json```

The accuracy results will be shown in the file **evaluation_results.json**