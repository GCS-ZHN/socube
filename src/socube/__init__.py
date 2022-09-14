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

__version__ = "1.1rc1"
__author__ = "Zhang.H.N"
__email__ = "zhang.h.n@foxmail.com"
__url__ = "https://github/GCS-ZHN/socube/"


def main(*args: str):
    import locale
    import json
    from argparse import ArgumentParser
    from importlib import resources
    zone, _ = locale.getdefaultlocale()
    try:
        with resources.open_text(__name__, "help.%s.json" % (zone)) as f:
            help = json.load(f)
    except FileNotFoundError:
        with resources.open_text(__name__, "help.en_US.json") as f:
            help = json.load(f)

    parser = ArgumentParser(help["command"])
    parser.add_argument("--version",
                        "-v",
                        action="store_true",
                        help=help["basic_args"]["version"])
    basic_args = parser.add_argument_group(help["basic_args"]["title"])
    basic_args.add_argument("--input",
                            "-i",
                            type=str,
                            default=None,
                            help=help["basic_args"]["input"])
    basic_args.add_argument("--output",
                            "-o",
                            default=None,
                            type=str,
                            help=help["basic_args"]["output"])
    basic_args.add_argument("--gpu-ids",
                            type=str,
                            default=None,
                            help=help["basic_args"]["gpu_ids"])
    basic_args.add_argument("--seed",
                            type=int,
                            default=None,
                            help=help["basic_args"]["seed"])
    basic_args.add_argument("--k",
                            type=int,
                            default=5,
                            help=help["basic_args"]["k"])
    basic_args.add_argument("--adj-factor",
                            type=float,
                            default=1.0,
                            help=help["basic_args"]["adj_factor"])
    basic_args.add_argument("--dim",
                            "-d",
                            type=int,
                            default=10,
                            help=help["basic_args"]["dim"])
    basic_args.add_argument("--cube-id",
                            type=str,
                            default=None,
                            help=help["basic_args"]["cube_id"])
    basic_args.add_argument("--only-embedding",
                            action="store_true",
                            default=False,
                            help=help["basic_args"]["only_embedding"]
    )

    model_args = parser.add_argument_group(help["model_args"]["title"])
    model_args.add_argument("--learning-rate",
                            "-lr",
                            type=float,
                            default=0.001,
                            help=help["model_args"]["learning_rate"])
    model_args.add_argument("--epochs",
                            "-e",
                            type=int,
                            default=100,
                            help=help["model_args"]["epochs"])
    model_args.add_argument("--train-batch-size",
                            type=int,
                            default=64,
                            help=help["model_args"]["train_batch_size"])
    model_args.add_argument("--vaild-batch-size",
                            type=int,
                            default=512,
                            help=help["model_args"]["valid_batch_size"])
    model_args.add_argument("--infer-batch-size",
                            type=int,
                            default=400,
                            help=help["model_args"]["infer_batch_size"])
    model_args.add_argument("--threshold",
                            "-t",
                            type=float,
                            default=0.5,
                            help=help["model_args"]["threshold"])
    model_args.add_argument("--enable-validation",
                            "-ev",
                            action="store_true",
                            help=help["model_args"]["enable_validation"])
    model_args.add_argument("--enable-multiprocess",
                            "-mp",
                            action="store_true",
                            help=help["model_args"]["enable_multiprocess"])

    notice_args = parser.add_argument_group(help["notice_args"]["title"])
    notice_args.add_argument("--mail",
                             type=str,
                             default=None,
                             help=help["notice_args"]["mail"])
    notice_args.add_argument("--mail-server",
                             type=str,
                             default=None,
                             help=help["notice_args"]["mail_server"])
    notice_args.add_argument("--mail-port",
                             type=int,
                             default=465,
                             help=help["notice_args"]["mail_port"])
    notice_args.add_argument("--mail-passwd",
                             type=str,
                             default=None,
                             help=help["notice_args"]["mail_passwd"])
    notice_args.add_argument("--enable-ssl",
                             action="store_true",
                             help=help["notice_args"]["enable_ssl"])
    args = parser.parse_args(args if args else None)

    if args.version:
        print(help["version"] % (__version__))
        return

    if args.input is None and args.cube_id is None:
        parser.print_help()
        return

    # Avoid import libraries at head
    from socube.utils.logging import getJobId, log
    log("Config", "Load required modules")
    import shutil
    import numpy as np
    import scanpy as sc
    import pandas as pd
    import torch

    from concurrent.futures import Future
    from socube.task.doublet import (createTrainData, checkData)
    from socube.data.preprocess import minmax, std
    from socube.utils.exception import ExceptionManager
    from socube.utils.mail import MailService
    from socube.utils.memory import parseGPUs
    from socube.cube import SoCube
    from socube.utils.io import mkDirs, writeCsv, writeHdf, checkExist
    from socube.utils.concurrence import ParallelManager
    from socube.data import filterData
    from socube.task.doublet import fit, infer, checkShape
    from os.path import dirname, join
    from scipy.sparse import issparse

    pd.options.mode.use_inf_as_na = True

    if None not in [
            args.mail, args.mail_server, args.mail_port, args.mail_passwd
    ]:
        log("Config", "Mail service inited for notification")
        mail_service = MailService(sender=args.mail,
                                   passwd=args.mail_passwd,
                                   server=args.mail_server,
                                   port=args.mail_port,
                                   ssl=args.enable_ssl)
    else:
        mail_service = None

    with ExceptionManager(mail_service=mail_service) as em:
        data_path = dirname(args.input) if args.input else "."
        home_path = args.output if args.output is not None else data_path
        cube_id = args.cube_id if args.cube_id else getJobId()
        embedding_path = join(home_path, "embedding", cube_id)
        train_path = join(embedding_path, "traindata")
        my_cube = SoCube.instance(map_type="pca",
                                  data_path=data_path,
                                  home_path=home_path)
        gpu_ids = None
        if torch.cuda.is_available():
            if args.gpu_ids is None:
                log(
                    "Config",
                    "Use CPU for training, but GPU is available, specify '--gpu-ids' to use GPU"
                )
            else:
                gpu_ids = parseGPUs(args.gpu_ids)
        else:
            if args.gpu_ids is not None:
                log(
                    "Config",
                    "GPU is not available, use CPU for training and ignore '--gpu-ids' setting"
                )

        # If cube id is specified, skip to repeat embedding fit
        if args.cube_id is None or not my_cube.load(embedding_path):
            with ParallelManager():
                if args.input is None:
                    log("Config", "No input file specified")
                    return

                checkExist(args.input)
                mkDirs(embedding_path)

                log("Preprocess", "Load data")
                label = None
                if args.input.endswith(".h5"):
                    samples = pd.read_hdf(args.input)

                elif args.input.endswith(".h5ad"):
                    samples = sc.read_h5ad(args.input)
                    if "type" in samples.obs:
                        label = (
                            samples.obs["type"] == "doublet").astype("int8")

                    samples = pd.DataFrame(samples.X.toarray() if issparse(
                        samples.X) else samples.X,
                                           index=samples.obs_names,
                                           columns=samples.var_names)

                else:
                    raise NotImplementedError("Unsupport file format")

                # if real label not specific, disable validation
                if label is None:
                    label = pd.Series(np.zeros_like(samples.index,
                                                    dtype="int8"),
                                      index=samples.index,
                                      name="type")
                    if args.enable_validation:
                        log(
                            "Config",
                            "Disable '--enable-validation' because no real label found"
                        )
                        args.enable_validation = False

                label.to_csv(join(embedding_path, "ExperimentLabel.csv"),
                             header=None)

                checkData(samples)
                if args.only_embedding:
                    train_data = samples
                else:
                    future: Future = createTrainData(samples,
                                                    output_path=embedding_path,
                                                    adj=args.adj_factor,
                                                    seed=args.seed)

                samples = samples.T
                writeHdf(
                    samples,
                    join(
                        embedding_path,
                        f"00-dataByGene[{samples.dtypes.iloc[0].name}][raw].h5"
                    ))

                log("Pre-processing", "Filter data")
                samples = filterData(samples)
                writeHdf(
                    samples,
                    join(
                        embedding_path,
                        "01-dataByGene[{:s}][filter].h5".format(
                            samples.dtypes.iloc[0].name)))

                my_cube.fit(samples,
                            device=gpu_ids[0] if isinstance(gpu_ids, list)
                            and len(gpu_ids) > 0 else "cpu",
                            seed=args.seed,
                            latent_dim=args.dim,
                            job_id=cube_id)
            
            if not args.only_embedding:
                train_data, train_label = future.result()

            log("Post-processing",
                "Processing data with log, std and feature minmax")
            train_data = minmax(std(np.log(train_data + 1)))
            log("Post-processing", "Single channels data is tranforming")
            my_cube.batchTransform(train_data, train_path)

            if not args.only_embedding:
                writeCsv(train_label,
                        join(train_path, "TrainLabel.csv"),
                        header=None)
                if checkExist(join(embedding_path, "ExperimentLabel.csv"),
                            raise_error=False):
                    shutil.copyfile(join(embedding_path, "ExperimentLabel.csv"),
                                    join(train_path, "ExperimentLabel.csv"))

        elif args.input is not None:
            log("Config", "input is ignored because cube id is specified")

        if args.only_embedding:
            return

        log("Train", "Data check before training start")
        dim = my_cube._config["latent_dim"]
        checkExist(join(train_path, "TrainLabel.csv"))
        checkShape(train_path, shape=(args.dim, None, None))
        log("Train", f"Data check passed, data path {train_path}")
        model_id = fit(home_dir=home_path,
                       data_dir=train_path,
                       lr=args.learning_rate,
                       gamma=0.99,
                       epochs=args.epochs,
                       train_batch=args.train_batch_size,
                       valid_batch=args.vaild_batch_size,
                       in_channels=dim,
                       transform=None,
                       shuffle=True,
                       gpu_ids=gpu_ids,
                       seed=args.seed,
                       label_file="TrainLabel.csv",
                       threshold=args.threshold,
                       k=args.k,
                       once=False,
                       use_index=False,
                       step=5,
                       max_acc_limit=1,
                       multi_process=args.enable_multiprocess)

        log("Inference", "Begin doublet detection output")
        infer(data_dir=train_path,
              home_dir=home_path,
              model_id=model_id,
              label_file="ExperimentLabel.csv",
              in_channels=dim,
              k=args.k,
              threshold=args.threshold,
              batch_size=args.infer_batch_size,
              gpu_ids=gpu_ids,
              with_eval=args.enable_validation,
              seed=args.seed,
              multi_process=args.enable_multiprocess)

        em.setNormalInfo("Doublet detection finished")
