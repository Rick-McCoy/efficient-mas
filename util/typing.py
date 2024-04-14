from jaxtyping import Bool, Float, Int
from torch import Tensor

AlignMaskTensor = Bool[Tensor, "batch codec text"]
AlignProbTensor = Float[Tensor, "batch codec text"]
AudioTensor = Float[Tensor, "batch audio"]
AudioChannelTensor = Float[Tensor, "batch 1 audio"]
AudioCodeTensor = Int[Tensor, "batch code codec"]
AudioEncTensor = Float[Tensor, "batch codec channel"]
AudioMaskTensor = Bool[Tensor, "batch codec"]
AudioSegmentTensor = Float[Tensor, "batch 1 segment"]
LengthTensor = Int[Tensor, "batch"]
LossTensor = Float[Tensor, ""]
LossBatchTensor = Float[Tensor, "batch"]
PhonemeTensor = Int[Tensor, "batch text"]
