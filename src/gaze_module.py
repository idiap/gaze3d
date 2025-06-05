# SPDX-FileCopyrightText: 2025 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Pierre Vuillecard  <pierre.vuillecard@idiap.ch>
#
# SPDX-License-Identifier: CC-BY-NC-4.0

import math
import os
from typing import Any, Dict, Tuple

import torch
import torch.optim as optim
from lightning import LightningModule
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MeanMetric, MinMetric

from src.data.components.base_dataset import DATASET_ID
from src.utils.metrics import AngularError, PredictionSave


class GazeModule(LightningModule):
    """
    A LightningModule to train, evaluate and predict 3D gaze.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        solver: DictConfig,
        loss: torch.nn.Module,
        compile: bool,
        output_path: str,
        pretrained_path: str = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.
        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # set paths
        self.output_path = output_path
        self.val_path = os.path.join(output_path, "metric", "val")
        os.makedirs(self.val_path, exist_ok=True)
        self.test_path = os.path.join(output_path, "metric", "test")
        os.makedirs(self.test_path, exist_ok=True)
        self.pred_path = os.path.join(output_path, "prediction")
        os.makedirs(self.pred_path, exist_ok=True)

        self.model = net
        if pretrained_path is not None:
            print(f"Loading pretrained model from {pretrained_path}")
            weight = torch.load(pretrained_path)["state_dict"]
            weight = {k: v for k, v in weight.items() if k.startswith("model")}
            weight = {k.replace("model.", ""): v for k, v in weight.items()}
            self.load_state_dict(weight)

        self.criterion = loss
        self.tasks = self.criterion.task_name
        metric_template = {
            "gaze": AngularError(),
        }

        self.train_angular = nn.ModuleDict(
            {task: metric_template[task].clone() for task in self.tasks}
        )

        val_datasets = [
            "all",
            "gaze360",
            "gaze360video",
            "gfie",
            "gfievideo",
            "mpsgaze",
            "gazefollow",
            "eyediap",
            "eyediapvideo",
            "vat",
            "vatvideo",
            "mpiiface",
        ]
        self.val_angular = nn.ModuleDict(
            {
                f"{val_data}_{task}": metric_template[task].clone()
                for task in self.tasks
                for val_data in val_datasets
            }
        )

        self.test_task = ["gaze"]

        test_datasets = [
            "gaze360",
            "gaze360video",
            "gfie",
            "gfievideo",
            "mpsgaze",
            "gazefollow",
            "eyediap",
            "eyediapvideo",
            "vat",
            "vatvideo",
            "mpiiface",
        ]
        self.test_angular = nn.ModuleDict(
            {
                f"{test_data}_{task}": metric_template[task].clone()
                for task in self.test_task
                for test_data in test_datasets
            }
        )

        self.test_pred = {task: PredictionSave() for task in self.test_task}

        # for averaging loss across batches
        tasks_loss = self.tasks + ["all"]
        self.train_loss = nn.ModuleDict({task: MeanMetric() for task in tasks_loss})
        self.val_loss = nn.ModuleDict({task: MeanMetric() for task in tasks_loss})
        self.test_loss = nn.ModuleDict({task: MeanMetric() for task in (self.test_task + ["all"])})

        # for tracking best so far validation accuracy
        self.val_ang_best = MinMetric()

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        for task in self.val_loss:
            self.val_loss[task].reset()
        for data_task in self.val_angular:
            self.val_angular[data_task].reset()

        self.val_ang_best.reset()

    def model_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        data_id = list(set(batch["data_id"].tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]

        output = self.model(batch["images"])
        loss = self.criterion(output, batch, data_id)

        return loss, output

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds = self.model_step(batch)

        batch_size = batch["images"].size(0)
        loss_sum = sum(loss.values())
        self.train_loss["all"](loss_sum)
        for task in loss:
            self.train_loss[task](loss[task])

        for task in loss:
            self.train_angular[task](preds[task], batch[task])

        for k in self.train_loss:
            self.log(
                f"train/loss_{k}",
                self.train_loss[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )
        for k in self.train_angular:
            self.log(
                f"train/angular_{k}",
                self.train_angular[k],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # return loss or backpropagation will fail
        return loss_sum

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds = self.model_step(batch)

        batch_size = batch["images"].size(0)
        loss_sum = sum(loss.values())
        self.val_loss["all"](loss_sum)
        self.log(
            f"val/loss_all",
            self.val_loss["all"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for task in loss:
            self.val_loss[task](loss[task])
            self.log(
                f"val/loss_{task}",
                self.val_loss[task],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        for task in loss:
            self.val_angular[f"all_{task}"](preds[task], batch[task])
            self.log(
                f"val/angular_all_{task}",
                self.val_angular[f"all_{task}"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        data_id = list(set(batch["data_id"].cpu().tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]
        for task in loss:
            self.val_angular[f"{DATASET_ID[data_id]}_{task}"](preds[task], batch[task])
            self.log(
                f"val/angular_{DATASET_ID[data_id]}_{task}",
                self.val_angular[f"{DATASET_ID[data_id]}_{task}"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_angular["all_gaze"].compute()  # get current val acc
        self.val_ang_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/angular_best", self.val_ang_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds = self.model_step(batch)

        batch_size = batch["images"].size(0)
        # update and log metrics
        loss_sum = sum(loss.values())
        self.test_loss["all"](loss_sum)
        self.log(
            "test/loss_all",
            self.test_loss["all"],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_size,
        )
        for task in self.test_task:
            self.test_loss[task](loss[task])
            self.log(
                f"test/loss_{task}",
                self.test_loss[task],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        data_id = list(set(batch["data_id"].cpu().tolist()))
        assert len(data_id) == 1
        data_id = data_id[0]

        if data_id in [1, 2, 3, 4, 6, 7, 8, 11, 12]:
            for task in self.test_task:
                self.test_pred[task].update(
                    preds[task],
                    batch[task],
                    batch["frame_id"],
                    batch["clip_id"],
                    batch["person_id"],
                    batch["data_id"],
                )

        for task in self.test_task:
            self.test_angular[f"{DATASET_ID[data_id]}_{task}"](preds[task], batch[task])
            self.log(
                f"test/angular_{DATASET_ID[data_id]}_{task}",
                self.test_angular[f"{DATASET_ID[data_id]}_{task}"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        for task in self.test_task:
            output_file_save = os.path.join(
                self.test_path, f"{task}_prediction_test_epoch_{self.current_epoch}.pkl"
            )

            save_prediction = self.test_pred[task].save(output_file_save)

            if task == "gaze":
                # TODO: add gaze metric computation
                # angular_results = compute_gaze_results(save_prediction)
                # for k, v in angular_results.items():
                #     self.log(f"test_datasets/{task}_{k}", v, on_step=False, on_epoch=True, prog_bar=False)
                # with open(
                #     os.path.join(
                #         self.test_path, f"{task}_metric_epoch_{self.current_epoch}.json"
                #     ),
                #     "w",
                # ) as f:
                #     json.dump(angular_results, f, indent=4)
                pass
            self.test_pred[task].reset()

    def predict_step(self, batch, batch_idx):
        # _, preds, targets = self.model_step(batch)
        preds = self.model(batch["images"])

        for task in self.test_task:
            pred_task = preds[task]
            gt = torch.zeros_like(pred_task)
            self.test_pred[task].update(
                preds[task],
                gt,
                batch["frame_id"],
                batch["clip_id"],
                batch["person_id"],
                batch["data_id"],
            )

    def on_predict_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        for task in self.test_task:
            output_file_save = os.path.join(
                self.pred_path, f"{task}_prediction_epoch_{self.current_epoch}.pkl"
            )

            save_prediction = self.test_pred[task].save(output_file_save)
            # save_pred_gaze_results(save_prediction, self.pred_path)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        # linear learning rate scaling for multi-gpu
        if (
            self.trainer.num_devices * self.trainer.num_nodes > 1
            and self.hparams.solver.apply_linear_scaling
        ):
            self.lr_scaler = (
                self.trainer.num_devices
                * self.trainer.num_nodes
                * self.trainer.accumulate_grad_batches
                * self.hparams.train_batch_size
                / 256
            )
        else:
            self.lr_scaler = 1

        optim_params = [
            {
                "params": filter(lambda p: p.requires_grad, self.trainer.model.parameters()),
                "lr": self.hparams.solver.lr * self.lr_scaler,
            }
        ]

        if self.hparams.solver.name == "AdamW":
            optimizer = optim.AdamW(
                params=optim_params,
                weight_decay=self.hparams.solver.weight_decay,
                betas=(0.9, 0.95),
            )
        elif self.hparams.solver.name == "Adam":
            optimizer = optim.Adam(
                params=optim_params,
                weight_decay=self.hparams.solver.weight_decay,
                betas=(0.9, 0.95),
            )
        elif self.hparams.solver.name == "SGD":
            optimizer = optim.SGD(
                params=optim_params,
                momentum=self.hparams.solver.momentum,
                weight_decay=self.hparams.solver.weight_decay,
            )
        else:
            raise NotImplementedError("Unknown solver : " + self.hparams.solver.name)

        def warm_start_and_cosine_annealing(epoch):
            lr_scale_min = 0.001
            if epoch < self.hparams.solver.warmup_epochs:
                lr = (epoch + 1) / self.hparams.solver.warmup_epochs
            else:
                lr = lr_scale_min + 0.5 * (1 - lr_scale_min) * (
                    1.0
                    + math.cos(
                        math.pi
                        * ((epoch + 1) - self.hparams.solver.warmup_epochs)
                        / (self.hparams.solver.max_epochs - self.hparams.solver.warmup_epochs)
                    )
                )
                #                                                        / (self.trainer.max_epochs - self.hparams.solver.warmup_epochs )))
            return lr

        if self.hparams.solver.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))],
                verbose=False,
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                self.hparams.solver.decay_steps,
                gamma=self.hparams.solver.decay_gamma,
                verbose=False,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
