# 🕰️ Conversation Chronicles

Code for Proceedings of EMNLP 2023 paper ["Conversation Chronicles: Towards Diverse Temporal and Relational Dynamics in Multi-Session Conversations"](https://arxiv.org/abs/2310.13420) Jihyoung Jang, MinSeong Boo, and Hyounghun Kim.

## Dataset and Model
- **We have uploaded our dataset and model to Hugging Face.**
- [Conversation Chronicles - Dataset](https://huggingface.co/datasets/jihyoung/ConversationChronicles)
- [ReBot - Generation Module](https://huggingface.co/jihyoung/rebot-generation)
- [ReBot - Summarizaiton Module](https://huggingface.co/jihyoung/rebot-summarization)

## Requirements 
```Shell
pip install torch
pip install transformers
pip install colorful
```
You can set up the environment easily through a **conda environment file**. Please refer to the usage section.

## Useage
```Shell
conda env create -f environment.yml
conda activate rebot
python rebot.py
```

## Acknowledgments
Base code is from ["SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization" paper's code repository](https://github.com/skywalker023/sodaverse), thanks.

## Citation Infromation
```
@inproceedings{jang-etal-2023-conversation,
    title = "Conversation Chronicles: Towards Diverse Temporal and Relational Dynamics in Multi-Session Conversations",
    author = "Jang, Jihyoung  and
      Boo, Minseong  and
      Kim, Hyounghun",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.838",
    doi = "10.18653/v1/2023.emnlp-main.838",
    pages = "13584--13606",
}
```
