import abc
import json
import os.path as osp
import time
from typing import Dict

import torch

from .utils.checkpoint import load_state_dict
from .utils.engine import setup_engine
from .utils.logger import get_logger
from .utils.misc import get_log_string
from .utils.parser import get_parser


def add_tester_arguments():
    parser = get_parser()
    parser.add_argument_group("tester", "tester arguments")
    parser.add_argument("--checkpoint", default=None, help="load from checkpoint")
    parser.add_argument("--test_epoch", type=int, default=None, help="test epoch")
    parser.add_argument("--cudnn_deterministic", type=bool, default=True, help="use deterministic method")


class BaseTester(abc.ABC):
    def __init__(self, cfg):
        # parser
        add_tester_arguments()
        parser = get_parser()
        self._args = parser.parse_args()
        self._cudnn_deterministic = self._args.cudnn_deterministic

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."

        # logger
        self._log_file = osp.join(cfg.exp.log_dir, "test-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
        self._logger = get_logger(log_file=self._log_file)

        # find checkpoint
        self._checkpoint = self._args.checkpoint
        if self._checkpoint is None and self._args.test_epoch is not None:
            self._checkpoint = f"epoch-{self._args.test_epoch}.pth"
        self._checkpoint = osp.join(cfg.exp.checkpoint_dir, self._checkpoint)
        assert self._checkpoint is not None, "Checkpoint is not specified."

        # print config
        self.log("Configs:\n" + json.dumps(cfg, indent=4))

        # initialize
        setup_engine(seed=cfg.exp.seed, cudnn_deterministic=self._cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        # data loader
        self.test_loader = None

    @property
    def args(self):
        return self._args

    @property
    def log_file(self):
        return self._log_file

    def load(self, filename):
        self.log('Loading from "{}".'.format(filename))
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        assert "model" in state_dict, "No model can be loaded."
        load_state_dict(self.model, state_dict["model"], strict=True)
        self.log("Model has been loaded.")
        if "metadata" in state_dict:
            epoch = state_dict["metadata"]["epoch"]
            total_steps = state_dict["metadata"]["total_steps"]
            self.log(f"Checkpoint metadata: epoch: {epoch}, total_steps: {total_steps}.")

    def register_model(self, model):
        """Register model."""
        model = model.cuda()
        self.model = model
        message = "Model description:\n" + str(model)
        self.log(message)
        return model

    def register_loader(self, test_loader):
        """Register data loader."""
        self.test_loader = test_loader

    def log(self, message, level="INFO"):
        self._logger.log(message, level=level)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    @abc.abstractmethod
    def test_step(self, iteration, data_dict) -> Dict:
        pass

    @abc.abstractmethod
    def eval_step(self, iteration, data_dict, output_dict) -> Dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self, summary_dict):
        pass

    def get_log_string(self, iteration, data_dict, output_dict, result_dict) -> str:
        return get_log_string(result_dict)

    @abc.abstractmethod
    def test_epoch(self):
        pass

    def run(self):
        assert self.test_loader is not None
        self.load(self._checkpoint)
        self.model.eval()
        torch.set_grad_enabled(False)
        self.test_epoch()
