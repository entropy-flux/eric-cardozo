from torch import zeros, concat 
from torch import Tensor   
from torch.nn import Module, Parameter, ModuleList
from torch.nn import Linear, LayerNorm, Conv2d
from torch.nn import Dropout 
from torch import tanh
from torch import exp, sigmoid  

class Perceptron(Module):
    def __init__(
        self,
        model_dimension: int,
        hidden_dimension: int,
        dropout_rate: float
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.q_projector = Linear(model_dimension, hidden_dimension) 
        self.v_projector = Linear(model_dimension, hidden_dimension)
        self.o_projector = Linear(hidden_dimension, model_dimension) 
        self.mu = Parameter(zeros(1, 1, 1))
        self.dropout = Dropout(dropout_rate)

    def forward(self, features: Tensor) -> Tensor:
        features = self.norm(features)
        q = self.q_projector(features)  
        v = self.v_projector(features)   
        features = sigmoid(q-self.mu) * tanh(q) * v
        features = self.dropout(features)
        features = self.o_projector(features)
        return self.dropout(features)

class Attention(Module): 
    def __init__(
        self, 
        model_dimension: int, 
        number_of_heads: int,
        dropout_rate: float
    ) -> None:
        super().__init__()
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.q_projector = Linear(model_dimension, model_dimension)
        self.k_projector = Linear(model_dimension, model_dimension)
        self.v_projector = Linear(model_dimension, model_dimension)  
        self.o_projector = Linear(model_dimension, model_dimension) 
        self.mu = Parameter(zeros(1, 1, 1, 1))
        self.scale = (model_dimension // number_of_heads) ** 0.5

        self.dropout = Dropout(dropout_rate) 
        self.number_of_heads = number_of_heads 

    def split(self, sequence: Tensor) -> Tensor:
        sequence = sequence.view(sequence.size(0), sequence.size(1), self.number_of_heads, sequence.size(-1) // self.number_of_heads)
        return sequence.transpose(1, 2)

    def merge(self, sequence: Tensor) -> Tensor: 
        return sequence.transpose(1, 2).reshape(sequence.size(0), sequence.size(2), -1)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.norm(sequence)
        q = self.q_projector(sequence)
        k = self.k_projector(sequence)
        v = self.v_projector(sequence)

        q, k, v = self.split(q), self.split(k), self.split(v)
        scores = q @ k.transpose(-2, -1) / self.scale
        scores = sigmoid(scores-self.mu) * exp(scores) 
        scores = scores /  scores.sum(dim=-1, keepdim=True)
        sequence = scores @ v

        sequence = self.o_projector(self.merge(sequence))
        return self.dropout(sequence)


class Encoder(Module): 
    def __init__(
        self, 
        model_dimension: int, 
        number_of_heads: int, 
        hidden_dimension: int, 
        dropout_rate: float
    ) -> None:
        super().__init__()
        self.attention = Attention(
            model_dimension,
            number_of_heads,
            dropout_rate=dropout_rate
        )
        self.reflection = Perceptron(
            model_dimension,
            hidden_dimension,
            dropout_rate=dropout_rate
        ) 

    def forward(self, sequence: Tensor) -> Tensor: 
        sequence = sequence + self.attention(sequence) 
        sequence = sequence + self.reflection(sequence) 
        return sequence


class Transformer(Module): 
    def __init__(
        self, 
        number_of_layers: int, 
        model_dimension: int, 
        number_of_heads: int, 
        hidden_dimension: int,
        dropout_rate: float
    ) -> None:
        super().__init__()
        self.encoders = ModuleList([
            Encoder(
                model_dimension,
                number_of_heads,
                hidden_dimension,
                dropout_rate
            )
            for _ in range(number_of_layers)
        ])

    def forward(self, sequence: Tensor) -> Tensor:
        for encoder in self.encoders:
            sequence = encoder(sequence)
        return sequence 

 
class Reservoir(Module):
    def __init__(
        self, 
        sequence_lenght: int,
        model_dimension: int
    )-> None:
        super().__init__()
        self.epsilon = Parameter(zeros(1, 1, model_dimension)) 
        self.delta = Parameter(zeros(1, sequence_lenght, model_dimension))

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence.flatten(2).transpose(1, 2) 
        return concat((self.epsilon.expand(sequence.size(0), -1, -1), sequence), dim=1) + self.delta


class GrandCanonicalViT(Module):  
    def __init__(
        self,  
        image_size: tuple[int, int], 
        patch_size: tuple[int, int],
        number_of_classes : int,
        model_dimension   : int,
        hidden_dimension  : int,
        number_of_heads   : int,
        number_of_layers  : int,
        number_of_channels: int,
        dropout_rate: float = 0.5
    )-> None:
        super().__init__()    
        self.image_size = image_size                 
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw    
 
        self.patcher = Conv2d(number_of_channels, model_dimension, kernel_size=(fh, fw), stride=(fh, fw))
        self.labeler = Reservoir(gh * gw + 1, model_dimension)  
          
        self.transformer = Transformer(
            number_of_layers,
            model_dimension,
            number_of_heads,
            hidden_dimension,
            dropout_rate
        )
 
        self.dropout = Dropout(dropout_rate)
        self.norm = LayerNorm(model_dimension, eps=1e-6) 
        self.head = Linear(model_dimension, number_of_classes)   
          
    def forward(self, sequence: Tensor) -> Tensor:  
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(0)
        sequence = self.patcher(sequence)  
        sequence = self.labeler(sequence)   
        sequence = self.dropout(sequence)
        sequence = self.transformer(sequence) 
        sequence = self.norm(sequence)[:, 0]  
        return self.head(sequence) 