# MIT License
#
# Copyright (c) 2022 Zhang.H.N
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from concurrent.futures import Future
from typing import List, Optional
import sklearn.metrics as metrics

import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
from socube.utils.concurrence import ParallelManager

from socube.utils.io import (checkExist, writeCsv, mkDirs, rm, loadTorchModule)
from socube.utils.memory import (GPUContextManager, visualBytes, autoClearIter)
from socube.utils.logging import (getJobId, log)
from socube.net import NetBase
from socube.train import (EarlyStopping, evaluateReport)
from .data import ConvClassifyDataset
from .model import SoCubeNet
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm.auto import tqdm
"""A module provided for model training and evaluation"""
__all__ = ["fit", "validate", "infer"]


def fit(home_dir: str,
        data_dir: str,
        lr: float = 0.001,
        gamma: float = 0.99,
        epochs: int = 100,
        train_batch: int = 32,
        valid_batch: int = 500,
        transform: nn.Module = None,
        in_channels: int = 10,
        num_workers: int = 0,
        shuffle: bool = False,
        seed: int = None,
        label_file: str = "label.csv",
        threshold: float = 0.5,
        k: int = 5,
        once: bool = False,
        use_index: bool = True,
        gpu_ids: List[str] = None,
        step: int = 5,
        model_id: str = None,
        pretrain_model_path: str = None,
        max_acc_limit: float = 1,
        multi_process: bool = False,
        **kwargs) -> str:
    """
    Train socube model.

    Parameters
    ----------
    home_dir: str
        the home directory of the specfic job
    data_dir: str
        the dataset's directory
    lr: float
        learning rate, default: 0.001
    gamma: float
        learning rate decay, default: 0.99
    epochs: int
        training epochs, default: 100
    train_batch: int
        training batch size, default: 32
    valid_batch: int
        validation batch size, default: 500
    transform: nn.Module
        sample transform, such as `Resize`
    in_channels: int
        the number of input channels, default: 10
    num_workers: int
        the number of workers for data loading, default: 0
    shuffle: bool
        if `True`, data will be shuffled while k-fold cross-valid
    seed: int
        random seed for k-fold cross-valid or sample
    label_file: str
        the label file csv name, default: "label.csv",
    threshold: float
        the threshold for classification, default: 0.5
    k: int
        k value of k-fold cross-valid
    once: bool
        if `True`, k-fold cross-validation runs first fold only
    use_index: bool
        If `True`, it will read sample file by index. Otherwise, it will read
        sample file by sample name.
    device_name: str
        the device name, default: "cpu"
    step: int
        the epoch step of learning rate decay, default: 5
    model_id: str
        the model id, If `None`, it will be generated automatically
    pretrain_model_path: str
        the pretrain model path, if not `None`, it will load the pretrain model
    max_acc_limit: float
        the max accuracy limit, if the accuracy is higher than this limit, the
        training will stop to prevent overfitting.
    multi_process: bool
        if `True`, it will use multi-process to train the model.
    **kwargs: dict
        the other parameters wanted to be saved in the log file.

    Returns
    ----------
    job id string
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    model_id = model_id if model_id else getJobId()
    log("Train", "Model ID: " + model_id)

    dataset = ConvClassifyDataset(data_dir=data_dir,
                                  transform=transform,
                                  labels=label_file,
                                  seed=seed,
                                  shuffle=shuffle,
                                  k=k,
                                  use_index=use_index)
    shape = dataset[0]["data"].shape
    mkDirs(f"{home_dir}/outputs/{model_id}")
    mkDirs(f"{home_dir}/models/{model_id}")
    mkDirs(f"{home_dir}/plots/{model_id}")
    report = pd.DataFrame()
    train_record = pd.DataFrame()
    train_kwargs = {
        "dataset": dataset,
        "train_batch": train_batch,
        "valid_batch": valid_batch,
        "in_channels": in_channels,
        "num_workers": num_workers,
        "shape": shape,
        "gamma": gamma,
        "lr": lr,
        "epochs": epochs,
        "home_dir": home_dir,
        "model_id": model_id,
        "pretrain_model_path": pretrain_model_path,
        "step": step,
        "max_acc_limit": max_acc_limit,
        "threshold": threshold
    }
    kwargs.update({
        "model": SoCubeNet.__name__,
        "kFold": k,
        "transform": transform,
        "data_dir": data_dir,
        "seed": seed,
        "label": label_file,
        "onlyOne": once,
        "multiProcess": multi_process,
        "gpu_ids": gpu_ids
    })
    kwargs.update(train_kwargs)
    config = pd.Series(kwargs)

    # Specific the default gpu device by context manager
    # If not, pytorch with use cuda:0 as default and load data to it when intializing,
    # no matter what device you want to use.
    # If free memory of cuda:0 is less than required, a Runtime Error will occurred
    # Otherwise, you will find TWO gpu device are used by your process in `nvidia-smi` report.
    
    with ParallelManager(paral_type="process", max_workers=k if multi_process else 1, verbose=True) as pm:
        results: List[Future] = []
        for fold, (train_set, valid_set) in enumerate(dataset.kFold, 1):
            device = torch.device("cpu")
            if gpu_ids is not None and len(gpu_ids) > 0 and torch.cuda.is_available():
                device = torch.device(gpu_ids[(fold-1)%len(gpu_ids)])
            results.append(pm.submit(_fit, fold = fold, device=device, train_set=train_set, valid_set = valid_set,  **train_kwargs))
            # quit k-fold
            if once:
                break

        for result in results:
            rep, train_rep = result.result()
            for head in rep:
                report[head] = rep[head]
            for head in train_rep:
                train_record[head] = train_rep[head]

    report["average"] = report.mean(axis=1)
    report["sample_stdev"] = report.std(axis=1)
    writeCsv(
        pd.concat([report["average"], config]),
        f"{home_dir}/outputs/{model_id}/{SoCubeNet.__name__}_aveReport.csv",
        header=None)

    if epochs > 0:
        writeCsv(
            train_record,
            f"{home_dir}/outputs/{model_id}/{SoCubeNet.__name__}_trainRecord.csv",
            index=None)

    writeCsv(
        report,
        f"{home_dir}/outputs/{model_id}/{SoCubeNet.__name__}_kFoldReport.csv")

    return model_id


def _fit(
    device: torch.device,
    train_set: Subset,
    valid_set: Subset,
    dataset: Dataset,
    train_batch: int,
    valid_batch: int,
    in_channels: int,
    num_workers: int,
    shape: tuple,
    gamma: float,
    lr: float,
    fold: int,
    epochs: int,
    home_dir: str,
    model_id: str,
    step: int,
    max_acc_limit: float,
    threshold: float,
    pretrain_model_path: Optional[str] = None
):
    """
    A internal function to train one fold of k-fold cross-validation.
    """
    log("Train", f"Train Fold {fold}")
    with torch.cuda.device(device if device.type == "cuda" else -1), GPUContextManager():
        data_loader = DataLoader(dataset=train_set,
                                sampler=dataset.sampler(train_set),
                                batch_size=train_batch,
                                num_workers=num_workers)
        valid_loader = DataLoader(dataset=valid_set,
                                batch_size=valid_batch,
                                sampler=dataset.sampler(valid_set))

        # init and load pretrained model
        model = SoCubeNet(in_channels,
                        dataset.typeCounts,
                        freeze=pretrain_model_path,
                        binary=True,
                        shape=shape).to(device)

        log("Train", f"Shape is {model.shape}")
        if pretrain_model_path:
            mp = pretrain_model_path % (fold)
            log("Train", "Loading pretained model %s" % (mp))
            loadTorchModule(model, mp)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=gamma)

        valid_loss, valid_acc, valid_label, valid_score = validate(
            data_loader=valid_loader, model=model, device=device)

        loop = tqdm(range(epochs))
        train_loss_array = np.full(epochs, -1, dtype=np.float32)
        valid_loss_array = np.full(epochs, -1, dtype=np.float32)
        train_acc_array = np.full(epochs, -1, dtype=np.float32)
        valid_acc_array = np.full(epochs, -1, dtype=np.float32)

        earlystop = EarlyStopping(
            path=f"{home_dir}/models/{model_id}/modelTmp_{fold}.pt",
            patience=10,
            verbose=10)

        for epoch in autoClearIter(loop):
            model.train()
            train_acc = 0
            train_loss = 0
            for index, batch in autoClearIter(enumerate(data_loader, 1)):
                data, label = batch.values()
                data = data.to(device)
                label = label.to(device)
                score = model(data)
                loss = model.criterion(score, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if model._binary:
                    predict = (score >= 0.5).cpu().long()
                else:
                    predict = score.argmax(1).cpu()
                train_acc += metrics.accuracy_score(label.cpu(), predict)
                train_loss += loss.item()
                loop.set_description("Fold %02d batch %03d/%03d" %
                                    (fold, index, len(data_loader)))
                loop.set_postfix(
                    loss=train_loss / index,
                    train_acc=train_acc / index,
                    valid_acc=valid_acc,
                    best_loss=earlystop._best,
                    pid=os.getpid(),
                    memo_res=visualBytes(
                        torch.cuda.memory_reserved(device)),
                    max_memo_res=visualBytes(
                        torch.cuda.max_memory_reserved(device)),
                    lr="%.06f" % (scheduler.get_last_lr()[0]))

                del data, label, score, loss

            if (epoch + 1) % step == 0:
                scheduler.step()

            valid_loss, valid_acc, valid_label, valid_score = validate(
                data_loader=valid_loader, model=model, device=device)

            if earlystop(valid_loss, model, valid_acc < max_acc_limit):
                best_label = valid_label
                best_score = valid_score

            train_acc_array[epoch] = train_acc / index
            valid_acc_array[epoch] = valid_acc
            train_loss_array[epoch] = train_loss / index
            valid_loss_array[epoch] = valid_loss
            del valid_label, valid_score
            if earlystop.earlyStop:
                log(__name__, "Early stopped for training")
                break

        loadTorchModule(model, earlystop._path)
        model.cpu()
        torch.save(
            model.state_dict(),
            f"{home_dir}/models/{model_id}/{SoCubeNet.__name__}_{fold}.pt")
        rm(earlystop._path)
        evaluate_result = evaluateReport(
            best_label, best_score,
            f"{home_dir}/plots/{model_id}/{SoCubeNet.__name__}_ROC_{fold}.png",
            f"{home_dir}/plots/{model_id}/{SoCubeNet.__name__}_PRC_{fold}.png",
            threshold=threshold
        )

        log("Train", "Best validate ACC: %.4f" % (evaluate_result["ACC"]))
        del model
        result = [{f"fold_{fold}": evaluate_result}]
        if epochs > 0:
            result.append({
                f"fold_{fold}_train_loss": train_loss_array,
                f"fold_{fold}_valid_loss": valid_loss_array,
                f"fold_{fold}_train_ACC": train_acc_array,
                f"fold_{fold}_valid_ACC": valid_acc_array
            })
        return result


@torch.no_grad()
def validate(data_loader: DataLoader,
             model: NetBase,
             device: torch.device,
             with_progress: bool = False) -> tuple:
    """
    Validate model performance basically

    Parameters
    ----------
    dataLoader: the torch dataloader object used for validation
    model: Network model implemented `NetBase`
        the model waited for validation
    device: the cpu/gpu device

    Returns
    ----------
    a quadra tuple of (average loss, average ACC, true label, predict score)
    """
    with torch.cuda.device(device if device.type == "cuda" else -1):
        model.to(device)
        model.eval()
        loss_ave = 0
        acc_ave = 0
        score_list = list()
        label_list = list()

        itererate = autoClearIter(enumerate(data_loader, 1))
        if with_progress:
            itererate = tqdm(itererate, desc="Validate")

        for index, batch in itererate:
            data, label = batch.values()
            data = data.to(device)
            label = label.to(device)
            score = model(data)
            loss = model.criterion(score, label)
            if model._binary:
                predict = (score >= 0.5).cpu().long()
            else:
                predict = score.argmax(1).cpu()

            acc = metrics.accuracy_score(label.cpu(), predict)
            loss_ave += loss.cpu().item()
            acc_ave += acc
            score_list.extend(score.cpu().numpy())
            label_list.extend(label.cpu().numpy())

            # destroy useless object and recycle GPU memory
            del data, label, loss, score

        loss_ave /= index
        acc_ave /= index

    return loss_ave, acc_ave, np.array(label_list), np.array(score_list)


def infer(data_dir: str,
          home_dir: str,
          model_id: str,
          label_file: str,
          in_channels: int = 10,
          k: int = 5,
          threshold: float = 0.5,
          batch_size: int = 400,
          gpu_ids: List[str] = None,
          with_eval: bool = False,
          seed: Optional[int] = None,
          multi_process: bool = False):
    """
    Model inference

    Parameters
    ----------
    data_dir: str
        the directory of data
    home_dir: str
        the home directory of output
    model_id: str
        the id of model
    label_file: str
        the label file used to inference
    in_channels: int
        the number of input channels
    k: int
        k value for k-fold cross validation
    threshold: float
        the threshold for binary classification
    batch_size: int
        the batch size for inference
    gpu_ids: List[str]
        the list of gpu ids
    with_eval: bool
        whether to evaluate the model performance
    seed: int
        the seed for random
    multi_process: bool
        whether to use multi-process for inference
    """
    dataset = ConvClassifyDataset(data_dir=data_dir,
                                  labels=label_file,
                                  shuffle=False,
                                  seed=seed,
                                  use_index=False)
    
    log("Inference",
        f"Current threshold: {threshold}, batch_size: {batch_size}")
    model_dir = os.path.join(home_dir, "models", model_id)
    checkExist(model_dir, types="dir")
    output_dir = os.path.join(home_dir, "outputs", model_id)

    plot_dir = os.path.join(home_dir, "plots", model_id)
    if with_eval:
        mkDirs(plot_dir)
    mkDirs(output_dir)
    ensemble_score_list = []
    with ParallelManager(max_workers=k if multi_process else 1, paral_type="process", verbose=True) as pm:
        results: List[Future] = []
        for fold in range(1, k + 1):
            device = torch.device("cpu")
            if gpu_ids is not None and len(gpu_ids) > 0 and torch.cuda.is_available():
                device = torch.device(gpu_ids[(fold-1)%len(gpu_ids)])
            results.append(pm.submit(_infer,
                in_channels,
                model_dir,
                dataset,
                batch_size,
                device,
                with_eval,
                plot_dir,
                threshold,
                fold,
                output_dir))
        for result in results:
            ensemble_score_list.append(result.result())
    log("Inference", "Model ensembling")
    ensemble_score = sum(ensemble_score_list) / k

    if with_eval:
        writeCsv(
            evaluateReport(
                label=dataset._labels.iloc[:, 0].values,
                score=ensemble_score,
                roc_plot_file=os.path.join(plot_dir, f"inference_roc_{threshold}.png"),
                prc_plot_file=os.path.join(plot_dir, f"inference_prc_{threshold}.png"),
                threshold=threshold),
            os.path.join(output_dir, f"inference_report_{threshold}.csv"))

    result = pd.DataFrame({"predict_score": ensemble_score},
                          index=dataset._labels.index)
    result["predict_type"] = result["predict_score"].apply(
        lambda s: "doublet" if s >= threshold else "singlet")
    writeCsv(result, os.path.join(output_dir, f"final_result_{threshold}.csv"))


def _infer(
    in_channels,
    model_dir,
    dataset,
    batch_size,
    device,
    with_eval,
    plot_dir,
    threshold,
    fold,
    output_dir):
    """
    Internal function for inference
    """
    model = SoCubeNet(in_channels=in_channels, out_channels=2, binary=True)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
    loadTorchModule(model,
                    os.path.join(model_dir, f"{SoCubeNet.__name__}_{fold}.pt"),
                    skipped=False)
    _, _, label, score = validate(dataloader, model, device, True)
    assert (dataset._labels.iloc[:, 0] == label).all()
    if with_eval:
        writeCsv(
            evaluateReport(
                label=label,
                score=score,
                roc_plot_file=os.path.join(plot_dir,
                                    f"inference_roc_{fold}_{threshold}.png"),
                prc_plot_file=os.path.join(plot_dir,
                                    f"inference_prc_{fold}_{threshold}.png"),
                threshold=threshold),
            os.path.join(output_dir, f"inference_report_{fold}_{threshold}.csv"))

    writeCsv(pd.DataFrame({"score": score}, index=dataset._labels.index),
                os.path.join(output_dir, f"inference_score_{fold}.csv"))
    return score
