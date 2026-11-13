"""Command entry point for RADDet's 3D RAD pretraining stage."""
import argparse
import json
from ..datasets.bev_dataset import BEVDataset
from .rad_pretraining import train_rad_backbone


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pretrain RADDet backbone with canonical normalized3D RAD annotations")
    parser.add_argument("--data",required=True,help="Version2 dataset root with boxes_rad on training and validation records")
    parser.add_argument("--out",required=True)
    parser.add_argument("--epochs",type=int,default=1)
    parser.add_argument("--batch-size",type=int,default=3)
    parser.add_argument("--max-steps",type=int)
    parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--resume",help="RAD pretraining last.pt; max-steps remains a total update bound")
    parser.add_argument("--allow-sensor-change",action="store_true",help="Explicitly acknowledge native sampling transfer; architecture shape must still match")
    args = parser.parse_args(argv)
    train = BEVDataset(args.data,"train",include_radar=True)
    validation = BEVDataset(args.data,"val",normalization=train.normalization,include_radar=True)
    checkpoint,history = train_rad_backbone(train,validation,args.out,epochs=args.epochs,
        batch_size=args.batch_size,max_steps=args.max_steps,learning_rate=args.learning_rate,
        device=args.device,seed=args.seed,num_threads=args.threads,resume=args.resume,
        allow_sensor_change=args.allow_sensor_change)
    print(json.dumps({"checkpoint":str(checkpoint),"last_epoch":history[-1],
                      "task_domain":"range_azimuth_doppler_not_cartesian","published_result_reproduced":False},indent=2))


if __name__ == "__main__":
    main()
