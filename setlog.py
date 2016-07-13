import os
import json
import logging.config

par_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)

def setup_logging(
    default_path=os.path.join(par_dir, 'docs', 'logging.json'), 
    default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        for handler in config['handlers']:
            if 'filename' in config['handlers'][handler]:
                config['handlers'][handler]['filename'] = os.path.join(
                    par_dir, config['handlers'][handler]['filename'])
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
