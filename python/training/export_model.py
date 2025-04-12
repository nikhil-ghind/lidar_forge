import argparse
import torch
from python.models.pointnet_generator import PointNetGenerator


def export(args):
    G = PointNetGenerator(condition_dim=64, num_points=args.num_points)
    state = torch.load(args.checkpoint, map_location="cpu")
    G.load_state_dict(state)
    G.eval()

    dummy = torch.zeros(1, 64)
    torch.onnx.export(
        G, dummy,
        args.output,
        input_names=["condition"],
        output_names=["point_cloud"],
        dynamic_axes={"condition": {0: "batch"}, "point_cloud": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_generator.pt")
    parser.add_argument("--output",     default="checkpoints/generator.onnx")
    parser.add_argument("--num_points", type=int, default=16384)
    export(parser.parse_args())
