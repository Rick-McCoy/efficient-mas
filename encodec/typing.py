from jaxtyping import Float
from torch import Tensor

ConvolutionalTensor = Float[Tensor, "batch channels time"]
LSTMTensor = Float[Tensor, "time batch channels"]
