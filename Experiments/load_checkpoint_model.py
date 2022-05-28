import torch
import curves

""" curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    args.fix_start,
    args.fix_end,
    architecture_kwargs=architecture.kwargs,
) """

base_model = None

checkpoint = torch.load("/tmp/curve/checkpoint-10.pt")
base_model.load_state_dict(checkpoint['model_state'])

""" model.import_base_parameters(base_model, k) """

pass
