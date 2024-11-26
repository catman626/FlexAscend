from mindspore import nn, ops

class Layer(nn.Cell):
    def __init__(self):
        super().__init__()

        self.layers = nn.SequentialCell(
            nn.Dense(in_channels=20, out_channels=30, has_bias=True), 
            nn.LayerNorm((30,)),
            nn.Dense(30, 20, has_bias=True)
        )

class Model(nn.Cell):
    def __init__(self):
        super().__init__()

        self.layers = nn.SequentialCell(
            Layer(), 
            Layer()
        )
        
if __name__ == "__main__":
    model = Model()
    for name, parameters in model.parameters_and_names():
        print(name)