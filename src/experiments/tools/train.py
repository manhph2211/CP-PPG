from comet_ml import Experiment
import sys
sys.path.append(".")
from src.trainer.engine import Trainer
from src.trainer.adverarial_engine import AdversarialTrainer
from src.utils.utils import get_config
from configs.seed import *


if __name__=="__main__":
    cfgs = get_config()

    with open('configs/experiment_apikey.txt','r') as f:
        api_key = f.read()

    tracking = Experiment(
        api_key = api_key,
        project_name = "PPG Data v2 - Window Based",
        workspace = "maxph2211",
    )
    tracking.log_parameters(cfgs)

    print("***********************************************")

    print(f"START TRAINING FOLD {cfgs['data']['fold']} ...")

    if cfgs['train']['adversarial']:
        print("USING ADVERSARIAL TRAINING ...")
        trainer = AdversarialTrainer(tracking, cfgs)
    else:
        print("USING NORMAL TRAINING ...")
        trainer = Trainer(tracking, cfgs)
        
    trainer.training_experiment()
    print("DONE!")

