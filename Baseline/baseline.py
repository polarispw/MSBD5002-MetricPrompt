"""
Here we implement 4 baseline introduced in the MetricPrompt paper on 3 dataset: ['agnews', 'dbpedia', 'yahoo']

ManualVerb: hand-crafted verbalizer
SoftVerb: WARP https://aclanthology.org/2021.acl-long.381.pdf
ProtoVerb: https://aclanthology.org/2022.acl-long.483.pdf
AVS: the author did not report any implementation details. But according to the cited paper(PET), the training
process is complex, involving many hyper-param settings and extra training datasets. So this maybe a mistake citing.
Besides, it performed worst among the methods reported in MetricPrompt.
Hence, we substitute it by Automatic Verbalizer (https://arxiv.org/pdf/2010.13641.pdf)

"""
import argparse
import os
import sys
import shutil

from datasets import load_from_disk, load_dataset
from openprompt.data_utils.data_processor import DataProcessor

sys.path.append(".")

from openprompt.trainer import ClassificationRunner, GenerationRunner
from openprompt.lm_bff_trainer import LMBFFClassificationRunner
from openprompt.protoverb_trainer import ProtoVerbClassificationRunner
from re import template
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader, check_config_conflicts
from openprompt.prompts import load_template, load_verbalizer
from openprompt.data_utils import FewShotSampler, InputExample
from openprompt.utils.logging import config_experiment_dir, init_logger
from openprompt.config import save_config_to_yaml, get_user_config, add_cfg_to_argparser, update_cfg_with_argparser
from openprompt.plms import load_plm_from_config


class AgnewsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [
            "World",
            "Sports",
            "Business",
            "Sci/Tech",
        ]
        self.id_cnt = 0

    def get_examples(self, data_dir, split):
        try:
            dataset = load_dataset('ag_news')
        except:
            dataset = load_from_disk(f"{data_dir}")

        if split == "valid" or split == "dev":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
        elif split == "train":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["train"]
        elif split == "test":
            dataset = dataset["test"]
        else:
            raise ValueError(f"split {split} not supported")

        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {"text": example["text"]}
        label = int(example['label'])
        guid = "{}".format(self.id_cnt)
        self.id_cnt += 1
        return InputExample(guid=guid, label=label, meta=meta)


class DbpediaProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [
            "Company",
            "EducationalInstitution",
            "Artist",
            "Athlete",
            "OfficeHolder",
            "MeanOfTransportation",
            "Building",
            "NaturalPlace",
            "Village",
            "Animal",
            "Plant",
            "Album",
            "Film",
            "WrittenWork",
        ]
        self.id_cnt = 0

    def get_examples(self, data_dir, split):
        try:
            dataset = load_dataset('dbpedia_14')
        except:
            dataset = load_from_disk(f"{data_dir}")

        if split == "valid" or split == "dev":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
        elif split == "train":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["train"]
        elif split == "test":
            dataset = dataset["test"]
        else:
            raise ValueError(f"split {split} not supported")

        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {"text": example["title"] + example["content"]}
        label = int(example['label'])
        guid = "{}".format(self.id_cnt)
        self.id_cnt += 1
        return InputExample(guid=guid, label=label, meta=meta)


class YahooAnswersTopicsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = [
            "Society & Culture",
            "Science & Mathematics",
            "Health",
            "Education & Reference",
            "Computers & Internet",
            "Sports",
            "Business & Finance",
            "Entertainment & Music",
            "Family & Relationships",
            "Politics & Government",
        ]

    def get_examples(self, data_dir, split):
        try:
            dataset = load_dataset('yahoo_answers_topics')
        except:
            dataset = load_from_disk(f"{data_dir}")

        if split == "valid" or split == "dev":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["test"]
        elif split == "train":
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)["train"]
        elif split == "test":
            dataset = dataset["test"]
        else:
            raise ValueError(f"split {split} not supported")

        return list(map(self.transform, dataset))

    def transform(self, example):
        meta = {"text": example["question_title"] + " " + example["question_content"]}
        label = int(example['topic'])
        guid = "{}".format(example["id"])
        return InputExample(guid=guid, label=label, meta=meta)


def get_ds(config, return_class=True, test=False):
    # loading from huggingface datasets
    print(f"loading dataset {config.dataset.name}...")
    ds_name = config.dataset.name
    if ds_name == "agnews":
        Processor = AgnewsProcessor()
    elif ds_name == "dbpedia":
        Processor = DbpediaProcessor()
    elif ds_name == "yahoo":
        Processor = YahooAnswersTopicsProcessor()
    else:
        raise ValueError(f"dataset {ds_name} not supported")

    train_ds = None
    valid_ds = None
    if not test:
        train_ds = Processor.get_train_examples()
        valid_ds = Processor.get_dev_examples()

    test_ds = Processor.get_test_examples()

    if return_class:
        return train_ds, valid_ds, test_ds, Processor
    else:
        return train_ds, valid_ds, test_ds


def build_dataloader(dataset, template, tokenizer, tokenizer_wrapper_class, config, split):
    dataloader = PromptDataLoader(
        dataset=dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size=config[split].batch_size,
        shuffle=config[split].shuffle_data,
        teacher_forcing=config[split].teacher_forcing if hasattr(config[split], 'teacher_forcing') else None,
        predict_eos_token=False,
        **config.dataloader
    )
    return dataloader


def main(config, args):
    # init logger, create log dir and set log level, etc.
    if args.resume and args.test:
        raise Exception("cannot use flag --resume and --test together")
    if args.resume or args.test:
        config.logging.path = EXP_PATH = args.resume or args.test
    else:
        EXP_PATH = config_experiment_dir(config)
        init_logger(os.path.join(EXP_PATH, "log.txt"), config.logging.file_level, config.logging.console_level)
        # save config to the logger directory
        save_config_to_yaml(config)

    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = get_ds(
        config,
        return_class=True,
        test=args.test is not None or config.learning_setting == 'zero_shot'
    )

    # main
    if config.learning_setting == 'full':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            resume=args.resume,
            test=args.test,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )
    elif config.learning_setting == 'few_shot':
        if config.few_shot.few_shot_sampling is None:
            raise ValueError("use few_shot setting but config.few_shot.few_shot_sampling is not specified")
        seeds = config.sampling_from_train.seed
        res = 0
        for seed in seeds:
            if not args.test:
                sampler = FewShotSampler(
                    num_examples_per_label=config.sampling_from_train.num_examples_per_label,
                    also_sample_dev=config.sampling_from_train.also_sample_dev,
                    num_examples_per_label_dev=config.sampling_from_train.num_examples_per_label_dev
                )
                train_sampled_dataset, valid_sampled_dataset = sampler(
                    train_dataset=train_dataset,
                    seed=seed
                )
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    resume=args.resume,
                    test=args.test,
                    train_dataset=train_sampled_dataset,
                    valid_dataset=valid_sampled_dataset,
                    test_dataset=test_dataset,
                )
            else:
                result = trainer(
                    os.path.join(EXP_PATH, f"seed-{seed}"),
                    config,
                    Processor,
                    test=args.test,
                    test_dataset=test_dataset,
                )
            res += result
            shutil.rmtree(os.path.join(EXP_PATH, f"seed-{seed}"))
        res /= len(seeds)
    elif config.learning_setting == 'zero_shot':
        res = trainer(
            EXP_PATH,
            config,
            Processor,
            zero=True,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            test_dataset=test_dataset,
        )

    print(f"#####    Final avg result: {res}      #####")


def trainer(EXP_PATH, config, Processor, train_dataset=None, valid_dataset=None, test_dataset=None, resume=None,
            test=None, zero=False):
    if not os.path.exists(EXP_PATH):
        os.mkdir(EXP_PATH)
    config.logging.path = EXP_PATH
    # set seed
    set_seed(config.reproduce.seed)

    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config, plm_wrapper_class = load_plm_from_config(config)

    # define template and verbalizer
    if config.task == "classification":
        # define prompt
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config,
                                     classes=Processor.labels)
        # load prompt’s pipeline model
        prompt_model = PromptForClassification(plm_model, template, verbalizer,
                                               freeze_plm=config.plm.optimize.freeze_para)

    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(plm_model, template, freeze_plm=config.plm.optimize.freeze_para,
                                           gen_config=config.generation)
    else:
        raise NotImplementedError(
            f"config.task {config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, plm_wrapper_class, config,
                                        "train") if train_dataset else None
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, plm_wrapper_class, config,
                                        "dev") if valid_dataset else None
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, plm_wrapper_class, config,
                                       "test") if test_dataset else None

    if config.task == "classification":
        if config.classification.auto_t or config.classification.auto_v:
            runner = LMBFFClassificationRunner(train_dataset=train_dataset,
                                               valid_dataset=valid_dataset,
                                               test_dataset=test_dataset,
                                               template=template,
                                               verbalizer=verbalizer,
                                               config=config
                                               )
        elif config.verbalizer == "proto_verbalizer":
            runner = ProtoVerbClassificationRunner(model=prompt_model,
                                                   train_dataloader=train_dataloader,
                                                   valid_dataloader=valid_dataloader,
                                                   test_dataloader=test_dataloader,
                                                   id2label=Processor.id2label,
                                                   config=config
                                                   )
        else:
            runner = ClassificationRunner(model=prompt_model,
                                          train_dataloader=train_dataloader,
                                          valid_dataloader=valid_dataloader,
                                          test_dataloader=test_dataloader,
                                          id2label=Processor.id2label,
                                          config=config
                                          )
    elif config.task == "generation":
        runner = GenerationRunner(
            model=prompt_model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=test_dataloader,
            config=config
        )
    else:
        raise NotImplementedError(
            f"config.task {config.task} is not implemented yet. Only classification and generation are supported.")

    if zero:
        res = runner.test()
    elif test:
        res = runner.test(ckpt='best')
    elif resume:
        res = runner.run(ckpt='last')
    else:
        res = runner.run()
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Global Config Argument Parser", allow_abbrev=False)
    parser.add_argument(
        "--config-yaml",
        default="scripts/agnews/proto_verb.yaml",
        help='the configuration file for this experiment.'
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=16,
        help='the number of training epochs.'
    )
    parser.add_argument(
        "--few-shot",
        type=int,
        default=4,
        help='the batch size for training.'
    )
    parser.add_argument(
        "--resume",
        type=str,
        help='a specified logging path to resume training.\
               It will fall back to run from initialization if no latest checkpoint are found.'
    )
    parser.add_argument(
        "--test",
        type=str,
        help='a specified logging path to test'
    )
    args, _ = parser.parse_known_args()
    config = get_user_config(args.config_yaml)

    config.train.num_epochs = args.train_epochs if args.train_epochs is not None else config.train.num_epochs
    config.sampling_from_train.num_examples_per_label = args.few_shot if args.few_shot is not None else config.sampling_from_train.num_examples_per_label
    config.train.batch_size = args.few_shot if args.few_shot is not None else config.train.batch_size

    config.sampling_from_train.num_examples_per_label_dev = 4 * args.few_shot if args.few_shot is not None else config.sampling_from_train.num_examples_per_label_dev
    config.dev.batch_size = 4 * args.few_shot if args.few_shot is not None else config.dev.batch_size

    add_cfg_to_argparser(config, parser)
    args = parser.parse_args()

    update_cfg_with_argparser(config, args)
    check_config_conflicts(config)

    os.path.exists(config.logging.path_base) or os.makedirs(config.logging.path_base)

    main(config, args)
