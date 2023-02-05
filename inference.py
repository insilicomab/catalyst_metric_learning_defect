"""
The inference results are saved to a csv file.
Usage:
    Inference with model on wandb:
        python inference.py \
        --timm_name {model name in timm} \
        --layer_name {layer name in catalyst}
        --wandb_run_path {wandb_run_path} \
        --image_size {image size default: 224}
        --embedding_size {embedder output size default: 512} \
        --k {top@k default: 10}
"""
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb

from src.dataset import ImageDataset, TestTransforms
from src.model import EncoderWithHead
from src.inference import predict_fn, InferenceModel


def main(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # read data
    train = pd.read_csv('input/train.csv')
    test = pd.read_csv('input/sample_submission.csv', header=None)

    # train image name list & label list
    image_name_list, label_list = train['id'], train['target']

    # test image name list & dummy label list
    x_test, dummy = test[0].values, test[0].values

    # index2target: key=index, value=target
    index2target = train['target'].to_dict()

    # dataset
    dataset = ImageDataset(
        image_name_list,
        label_list,
        img_dir='input/train_data',
        transform=TestTransforms(image_size=args.image_size),
        phase='test'
    )

    # test dataset
    test_dataset = ImageDataset(
        x_test,
        dummy,
        img_dir='input/test_data',
        transform=TestTransforms(image_size=args.image_size),
        phase='test'
    )

    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # model
    model = EncoderWithHead(
        model_name=args.timm_name,
        pretrained=False,
        layer_name=args.layer_name,
        embedding_size=args.embedding_size,
        num_classes=args.num_classes,
    )

    # restore model in wandb
    best_model_weights = wandb.restore('model_best.pth', run_path=args.wandb_run_path)

    model.load_state_dict(torch.load(best_model_weights.name, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    # inference
    im = InferenceModel(model.encoder)
    im.train_knn(dataset)

    df, df_top1, df_mode = predict_fn(
        inference_model=im,
        test_dataloader=test_dataloader,
        index_to_target=index2target,
        k=args.k
    )

    df.to_csv('submit/inference.csv', sep=',', index=None)
    df_top1.to_csv('submit/submission_top1.csv', sep=',', header=None, index=None)
    df_mode.to_csv('submit/submission_mode.csv', sep=',', header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_name', type=str)
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--wandb_run_path', type=str)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--embedding_size', type=int, default=512)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()

    main(args)