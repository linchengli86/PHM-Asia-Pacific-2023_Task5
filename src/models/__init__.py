from .encoder import build_encoder
from .heads import MultiTaskHeads
from .dual_channel import DualChannelModel
from .prototypical import ProtoTaskHeads, ProtoLoss, build_prototypes, compute_proto_logits, EMAPrototypes
from .mixup import mixup_data, manifold_mixup, mixup_criterion
from .distill import DistillationLoss, TaskDistiller
