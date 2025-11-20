# LLaDA-Rec

This repository contains the official implementation of the paper  
**“LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation.”**

## 📦 Dataset

We use the **Amazon Reviews 2023** dataset. You may download it from the [Amazon 2023 Dataset Page](https://amazon-reviews-2023.github.io/index.html).

Below are the direct download links for the three categories used in our experiments:

- **Industrial_and_Scientific**  
  [train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Industrial_and_Scientific.train.csv.gz) / [valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Industrial_and_Scientific.valid.csv.gz) / [test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.test.csv.gz/) / [meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Industrial_and_Scientific.jsonl.gz)

- **Musical_Instruments**  
  [train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.train.csv.gz) / [valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.valid.csv.gz) / [test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Musical_Instruments.test.csv.gz) / [meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Musical_Instruments.jsonl.gz)

- **Video_Games**  
  [train](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.train.csv.gz) / [valid](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.valid.csv.gz) / [test](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/last_out_w_his/Video_Games.test.csv.gz) / [meta](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz)

After downloading, place the files under the following directory structure:
```bash
./cache
├── AmazonReviews2023
│   ├── Industrial_and_Scientific
│   │   ├── raw
│   │   │   ├── meta_Industrial_and_Scientific.jsonl
│   │   │   ├── Industrial_and_Scientific.train.csv
│   │   │   ├── Industrial_and_Scientific.valid.csv
│   │   │   └── Industrial_and_Scientific.test.csv
│   ├── Musical_Instruments
│   │   ├── raw
│   │   │   ├── meta_Musical_Instruments.jsonl
│   │   │   ├── Musical_Instruments.train.csv
│   │   │   ├── Musical_Instruments.valid.csv
│   │   │   └── Musical_Instruments.test.csv
│   ├── Video_Games
│   │   ├── raw
│   │   │   ├── meta_Video_Games.jsonl
│   │   │   ├── Video_Games.train.csv
│   │   │   ├── Video_Games.valid.csv
│   │   │   └── Video_Games.test.csv
```


## 🚀 Training Pipeline

In `run_pipeline.py`, specify the dataset category:

```python
category = "Industrial_and_Scientific"  # or "Musical_Instruments", "Video_Games"
```
Then run:

```bash
python run_pipeline.py
```

## 📭 Contact

If you have any questions, feel free to contact: [shiteng@ruc.edu.cn](mailto:shiteng@ruc.edu.cn)

## 📚 Citation

If you find this repository useful, please cite:

```bibtex
@article{shi2025llada,
  title={LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation},
  author={Shi, Teng and Shen, Chenglei and Yu, Weijie and Nie, Shen and Li, Chongxuan and Zhang, Xiao and He, Ming and Han, Yan and Xu, Jun},
  journal={arXiv preprint arXiv:2511.06254},
  year={2025}
}
```
