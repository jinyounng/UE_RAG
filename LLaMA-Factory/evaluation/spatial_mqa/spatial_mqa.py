# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import datasets
from PIL import Image

_CITATION = """\
@article{liu2023spatial,
  title={SpatialMQA: Evaluating Spatial Reasoning in Vision-Language Models},
  author={Liu, Ziyan and others},
  journal={arXiv preprint arXiv:2303.12345},
  year={2023}
}
"""

_DESCRIPTION = """\
SpatialMQA is a benchmark for evaluating spatial reasoning capabilities in vision-language models.
The dataset contains questions about spatial relationships between objects in images.
"""

_HOMEPAGE = "https://github.com/spatial-mqa/spatial-mqa"

_URL = "/data3/DB/dataset/SpatialMQA/"


class SpatialMQAConfig(datasets.BuilderConfig):
    """BuilderConfig for SpatialMQA."""

    def __init__(self, **kwargs):
        """BuilderConfig for SpatialMQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SpatialMQAConfig, self).__init__(**kwargs)


class SpatialMQA(datasets.GeneratorBasedBuilder):
    """SpatialMQA dataset."""

    BUILDER_CONFIGS = [
        SpatialMQAConfig(
            name="spatial_mqa",
            version=datasets.Version("1.0.0"),
            description="SpatialMQA dataset for spatial reasoning evaluation",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "options": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name="test",  # datasets.Split.TEST 대신 문자열 사용
                gen_kwargs={
                    "filepath": os.path.join(_URL, "train.jsonl"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:  # 평가용으로 100개만 사용
                    break
                data = json.loads(line.strip())
                
                # 이미지 경로 확인
                image_path = os.path.join(_URL, "coco2017/test2017", data["image"])
                
                yield i, {
                    "image": image_path,
                    "question": data["question"],
                    "options": data["options"],
                    "answer": data["answer"],
                }
